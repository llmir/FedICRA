# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from flwr.common import (GetParametersRes, Status, FitRes, EvaluateRes, ndarrays_to_parameters,
                        parameters_to_ndarrays, GetPropertiesIns, GetPropertiesRes)
from flwr.common.logger import log
from flwr.server.server import Server, fit_clients, evaluate_clients
from flwr.server.history import History
from flwr.server.strategy.strategy import Strategy
from flwr.server.strategy.fedavg import FedAvg
from flwr.server.strategy.fedadagrad import FedAdagrad
from flwr.server.strategy.fedadam import FedAdam
from flwr.server.strategy.fedyogi import FedYogi
from collections import OrderedDict
from logging import DEBUG, INFO, WARNING
from torchvision.utils import make_grid
from tqdm import tqdm


import timeit
import copy
from functools import reduce
import math
from torch.cuda.amp import autocast, GradScaler

from networks.net_factory import net_factory
from val_2D import test_single_volume, test_single_volume_ds
from utils.TreeEnergyLoss.kernels.lib_tree_filter.modules.tree_filter import MinimumSpanningTree
from utils.TreeEnergyLoss.kernels.lib_tree_filter.modules.tree_filter import TreeFilter2D



class BaseClient(fl.client.Client):

    def __init__(self, args, model, trainloader, valloader):
        self.args = args
        self.cid = args.cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.current_iter = 0
        self.current_lr = self.args.base_lr
        self.sampled_batches = []
        self.properties = {'cid': self.cid}

    def get_parameters(self, ins):
        print('Client {}: get_parameters'.format(self.cid))
        weights = self.model.get_weights(ins.config)
        parameters = fl.common.ndarrays_to_parameters(weights)
        return GetParametersRes(parameters=parameters, status=Status('OK', 'Success'))

    def get_properties(self, ins):
        print('Client {}: get_properties'.format(self.cid))
        return GetPropertiesRes(properties=self.properties, status=Status('OK', 'Success'))

    def fit(self, ins):
        print('Client {}: fit'.format(self.cid))

        weights = fl.common.parameters_to_ndarrays(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        self.model.set_weights(weights, config)
        loss, metrics_ = self._train(config)

        weights_prime = self.model.get_weights(config)
        params_prime = fl.common.ndarrays_to_parameters(weights_prime)
        num_examples_train = len(self.trainloader)
        fit_duration = timeit.default_timer() - fit_begin
        metrics_['fit_duration'] = fit_duration

        return FitRes(
            status=Status('OK', 'Success'),
            parameters=params_prime,
            num_examples=num_examples_train,
            metrics=metrics_
        )

    def evaluate(self, ins):
        print('Client {}: evaluate'.format(self.cid))

        weights = fl.common.parameters_to_ndarrays(ins.parameters)
        config = ins.config

        self.model.set_weights(weights, config)
        loss, metrics_ = self._validate(config)

        return EvaluateRes(
            status=Status('OK', 'Success'),
            loss=loss,
            num_examples=len(self.valloader),
            metrics=metrics_
        )

    def _train(self, config):
        raise NotImplementedError

    def _validate(self, config):
        self.model.eval()
        val_metrics = evaluate(self.args, self.model, self.valloader, self.amp)

        if val_metrics['val_mean_dice'] > self.best_performance:
            self.best_performance = val_metrics['val_mean_dice']
            state_dict = self.model.model.state_dict()
            save_mode_path = os.path.join(self.args.snapshot_path, 'client_{}_async_iter_{}_dice_{}.pth'.format(
                                        self.cid, self.current_iter, round(self.best_performance, 4)))
            save_best = os.path.join(self.args.snapshot_path, 'client_{}_async_{}_best_model.pth'.format(self.cid, self.args.model))
            torch.save(state_dict, save_mode_path)
            torch.save(state_dict, save_best)
            log(INFO, 'save model to {}'.format(save_mode_path))

        val_metrics = { 'client_{}_{}'.format(self.cid, k): v for k, v in val_metrics.items() }

        return 0.0, val_metrics


VAL_METRICS = ['dice', 'hd95', 'recall', 'precision', 'jc', 'specificity', 'ravd']
def evaluate(args, model, dataloader, amp=False):
    metric_list = 0.0
    metrics_ = {}
    for i_batch, sampled_batch in enumerate(dataloader):
        metric_i = test_single_volume(
            sampled_batch['image'], sampled_batch['label'], model, classes=args.num_classes, amp=amp)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(dataloader.dataset)
    for class_i in range(args.num_classes-1):
        for metric_i, metric_name in enumerate(VAL_METRICS):
            metrics_['val_{}_{}'.format(class_i+1, metric_name)] = metric_list[class_i, metric_i]

    for metric_i, metric_name in enumerate(VAL_METRICS):
        metrics_['val_mean_{}'.format(metric_name)] = np.mean(metric_list, axis=0)[metric_i]
    return metrics_


def get_evaluate_fn(args, valloader, amp=False):
    def evaluate_fn(server_round, weights, place):
        model = net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes)
        state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)
        })
        model.load_state_dict(state_dict, strict=True)
        model.cuda()
        model.eval()
        metrics_ = evaluate(args, model, valloader, amp)
        return 0.0, metrics_

    return evaluate_fn


import random
def evaluate_uncertainty(args, model, dataloader, amp=False):
    uncertainty_list = []
    for i_batch, sampled_batch in enumerate(dataloader):
        if args.img_class == 'faz':
            volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        elif args.img_class == 'odoc' or args.img_class == 'polyp':
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

        with autocast(enabled=amp):
            rot_times = random.randrange(0, 4)
            rotated_volume_batch = torch.rot90(volume_batch, rot_times, [2, 3])
            T = 8
            _, _, w, h = volume_batch.shape
            volume_batch_r = rotated_volume_batch.repeat(2, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, args.num_classes, w, h]).cuda()
            for i in range(T // 2):
                ema_inputs = volume_batch_r + \
                    torch.clamp(torch.randn_like(
                        volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride *
                          (i + 1)] = model(ema_inputs)[0]
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, args.num_classes, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * \
                torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            uncertainty_list.append(torch.mean(uncertainty).item())

    overall_uncertainty = np.mean(uncertainty_list)
    return overall_uncertainty


class MyServer(Server):

    def __init__(self, args, writer, state_dict_keys, train_scalar_metrics, train_image_metrics, val_metrics, client_manager, strategy):
        super(MyServer, self).__init__(client_manager=client_manager, strategy=strategy)
        self.args = args
        self.writer = writer
        self.state_dict_keys = state_dict_keys
        self.train_scalar_metrics = train_scalar_metrics
        self.train_image_metrics = train_image_metrics
        self.val_metrics = val_metrics

    # pylint: disable=too-many-locals
    def fit(self, num_rounds, timeout):
        '''Run federated averaging for a number of rounds.'''
        history = History()

        # Initialize parameters
        log(INFO, 'Initializing global parameters')
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, 'Evaluating initial parameters')
        res = self.strategy.evaluate(0, parameters=self.parameters)
        print(res)
        if res is not None:
            log(
                INFO,
                'initial parameters (loss, other metrics): %s, %s',
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, 'FL starting')
        start_time = timeit.default_timer()

        num_classes = self.args.num_classes
        max_iterations = self.args.max_iterations
        snapshot_path = self.args.snapshot_path
        iters = self.args.iters
        min_num_clients = self.args.min_num_clients
        client_id_list = range(min_num_clients)
 
        if len(self.train_image_metrics * 2) > 6:
            nrow = len(self.train_image_metrics)
        else:
            nrow = len(self.train_image_metrics) * 2

        def parameters_to_state_dict(parameters):
            weights = fl.common.parameters_to_ndarrays(parameters)
            state_dict = OrderedDict(
                {k: torch.tensor(v) for k, v in zip(self.state_dict_keys, weights)}
            )
            return state_dict

        def get_client_state_dict(central_parameters, client_parameters):
            central_state_dict = parameters_to_state_dict(central_parameters)
            client_state_dict = parameters_to_state_dict(client_parameters)

            local_keys = []
            for key in self.state_dict_keys:
                # print(key)
                if key not in local_keys:
                    client_state_dict[key] = central_state_dict[key]
            return client_state_dict

        best_performance = 0.0
        iterator = tqdm(range(iters, num_rounds+iters, iters), ncols=70)
        for current_round in iterator:
            iter_num = current_round
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit[0] is None:
                log(INFO, 'round {}: fit failed'.format(current_round))
                continue

            parameters_prime, metrics_prime, (results_prime, failtures_prime) = res_fit
            self.parameters = parameters_prime
            images = []
            for client_id in client_id_list:
                for metric_name in self.train_scalar_metrics:
                    self.writer.add_scalar('info/client_{}_{}'.format(client_id, metric_name), metrics_prime['client_{}_{}'.format(client_id, metric_name)], iter_num)
                for metric_name in self.train_image_metrics:
                    images.append(fl.common.bytes_to_ndarray(metrics_prime['client_{}_{}'.format(client_id, metric_name)]))

                if self.args.strategy in ['FedICRA']:
                    self.writer.add_scalar('info/client_{}_loss_lc'.format(client_id), metrics_prime['client_{}_loss_lc'.format(client_id)], iter_num)

            self.writer.add_image(
                'train/grid_image',
                make_grid(torch.tensor(np.array(images)), nrow=nrow),
                iter_num
            )

            # Evaluate model using strategy implementation
            if iter_num > 0 and iter_num % self.args.eval_iters == 0:

                if self.args.strategy not in PERSONALIZED_FL:
                    res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
                    if res_cen is None:
                        log(INFO, 'round {}: evaluate failed'.format(current_round))
                        continue
                    loss_cen, metrics_cen = res_cen
                    log(
                        INFO,
                        'fit progress: (%s, %s, %s, %s)',
                        current_round,
                        loss_cen,
                        metrics_cen,
                        timeit.default_timer() - start_time,
                    )

                res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
                if res_fed[0] is None:
                    log(INFO, 'round {}: evaluate failed'.format(current_round))
                    continue
                loss_fed, evaluate_metrics_fed, (results_fed, failtures_fed) = res_fed
                # print(loss_fed, evaluate_metrics_fed.keys())
                for client_id in client_id_list:
                    for class_i in range(num_classes-1):
                        for metric_name in self.val_metrics:
                            self.writer.add_scalar('info_client_{}/val_{}_{}'.format(client_id, class_i+1, metric_name),
                                                evaluate_metrics_fed['client_{}_val_{}_{}'.format(client_id, class_i+1, metric_name)], iter_num)
                    for metric_name in self.val_metrics:
                        self.writer.add_scalar('info_client_{}/val_mean_{}'.format(client_id, metric_name),
                                                evaluate_metrics_fed['client_{}_val_mean_{}'.format(client_id, metric_name)], iter_num)
                    self.writer.add_scalar('info/client_{}_val_mean_dice'.format(client_id),
                                        evaluate_metrics_fed['client_{}_val_mean_dice'.format(client_id)], iter_num)

                # print(evaluate_metrics_fed.keys())

                if self.args.strategy not in PERSONALIZED_FL:
                    mean_metrics = metrics_cen
                else:
                    mean_metrics = evaluate_metrics_fed
                # print(mean_metrics.keys())

                for class_i in range(num_classes-1):
                    for metric_name in self.val_metrics:
                        self.writer.add_scalar('info/val_{}_{}'.format(class_i+1, metric_name), mean_metrics['val_{}_{}'.format(class_i+1, metric_name)], iter_num)

                metric_log = 'iteration {} : '.format(iter_num)
                for metric_name in self.val_metrics:
                    metric_log += 'mean_{} : {}; '.format(metric_name, mean_metrics['val_mean_{}'.format(metric_name)])
                    self.writer.add_scalar('info/val_mean_{}'.format(metric_name), mean_metrics['val_mean_{}'.format(metric_name)], iter_num)
                    self.writer.add_scalar('info/val_avg_mean_{}'.format(metric_name), evaluate_metrics_fed['val_avg_mean_{}'.format(metric_name)], iter_num)

                val_mean_dice = mean_metrics['val_mean_dice']
                log(INFO, metric_log)

                if val_mean_dice > best_performance:
                    best_performance = val_mean_dice
                    if self.args.strategy not in PERSONALIZED_FL:
                        state_dict = parameters_to_state_dict(self.parameters)
                        save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(
                                                    iter_num, round(best_performance, 4)))
                        save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(self.args.model))
                        torch.save(state_dict, save_mode_path)
                        torch.save(state_dict, save_best)
                        log(INFO, 'save model to {}'.format(save_mode_path))

                    for client_id in client_id_list:
                        first_metric_name = 'client_{}_{}'.format(client_id, self.train_scalar_metrics[0])
                        for _, fit_res in results_prime:
                            # print(client_id, first_metric_name in fit_res.metrics.keys())
                            if first_metric_name in fit_res.metrics.keys():
                                client_state_dict = get_client_state_dict(self.parameters, fit_res.parameters)
                                client_save_mode_path = os.path.join(snapshot_path, 'client_{}_iter_{}_dice_{}.pth'.format(
                                    client_id, iter_num, round(evaluate_metrics_fed['client_{}_val_mean_dice'.format(client_id)], 4)
                                ))
                                client_save_best = os.path.join(snapshot_path, 'client_{}_{}_best_model.pth'.format(
                                                                client_id, self.args.model))
                                torch.save(client_state_dict, client_save_mode_path)
                                torch.save(client_state_dict, client_save_best)
                                log(INFO, 'save model to {}'.format(client_save_mode_path))

            if iter_num > 0 and iter_num % 3000 == 0:
                if self.args.strategy not in PERSONALIZED_FL:
                    state_dict = parameters_to_state_dict(self.parameters)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}.pth'.format(iter_num))
                    torch.save(state_dict, save_mode_path)
                    log(INFO, 'save model to {}'.format(save_mode_path))

                for client_id in client_id_list:
                    first_metric_name = 'client_{}_{}'.format(client_id, self.train_scalar_metrics[0])
                    for _, fit_res in results_prime:
                        if first_metric_name in fit_res.metrics.keys():
                            client_state_dict = get_client_state_dict(self.parameters, fit_res.parameters)
                            client_save_mode_path = os.path.join(snapshot_path, 'client_{}_iter_{}.pth'.format(client_id, iter_num))
                            torch.save(client_state_dict, client_save_mode_path)
                            log(INFO, 'save model to {}'.format(client_save_mode_path))

            if iter_num >= max_iterations:
                break

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, 'FL finished in %s', elapsed)
        return history


def fit_metrics_aggregation_fn(fit_metrics):
    metrics = { k: v for _, client_metrics in fit_metrics for k, v in client_metrics.items() }
    return metrics


def get_evaluate_metrics_aggregation_fn(args, val_metrics):
    def evaluate_metrics_aggregation_fn(evaluate_metrics):
        metrics = { k: v for _, client_metrics in evaluate_metrics for k, v in client_metrics.items() }
        weights = {}
        for client_id in range(args.min_num_clients):
            first_metric_name = 'client_{}_val_mean_{}'.format(client_id, val_metrics[0])
            for client_num_examples, client_metrics in evaluate_metrics:
                if first_metric_name in client_metrics.keys():
                    weights['client_{}'.format(client_id)] = client_num_examples
        # print(weights)

        def weighted_metric(metric_name):
            num_total_examples = sum([client_num_examples for client_num_examples in weights.values()])
            weighted_metric = [weights['client_{}'.format(client_id)] * metrics['client_{}_{}'.format(client_id, metric_name)]
                                for client_id in range(args.min_num_clients)]
            return sum(weighted_metric) / num_total_examples

        def mean_metric(metric_name):
            return np.mean([metrics['client_{}_{}'.format(client_id, metric_name)]
                            for client_id in range(args.min_num_clients)])

        metrics.update({'val_{}_{}'.format(class_i+1, metric_name): weighted_metric('val_{}_{}'.format(class_i+1, metric_name))
                        for class_i in range(args.num_classes-1) for metric_name in val_metrics})
        metrics.update({'val_mean_{}'.format(metric_name): weighted_metric('val_mean_{}'.format(metric_name))
                        for metric_name in val_metrics})
        metrics.update({'val_avg_mean_{}'.format(metric_name): mean_metric('val_mean_{}'.format(metric_name))
                        for metric_name in val_metrics})

        return metrics

    return evaluate_metrics_aggregation_fn


PERSONALIZED_FL = ['FedICRA']
CENTRALIZED_FL = ['FedAvg', 'FedAdagrad', 'FedAdam', 'FedYogi']
def get_strategy(name, **kwargs):
    assert name in (CENTRALIZED_FL + PERSONALIZED_FL)
    if name == 'FedAvg':
        strategy = FedAvg(**kwargs)
    elif name == 'FedAdagrad':
        strategy = FedAdagrad(**kwargs)
    elif name == 'FedAdam':
        strategy = FedAdam(**kwargs)
    elif name == 'FedYogi':
        strategy = FedYogi(**kwargs)
    elif name == 'FedICRA':
        strategy = FedICRA(**kwargs)
    else:
        raise NotImplementedError

    return strategy


class FedICRA(FedAvg):

    def __repr__(self) -> str:
        rep = f"FedICRA(accept_failures={self.accept_failures})"
        return rep


class MyModel(nn.Module):

    def __init__(self, args, model, trainloader, valloader):
        super(MyModel, self).__init__()
        self.args = args
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.amp=(args.amp == 1)
        if self.amp:
            self.scaler = GradScaler()

        num_params = 0
        for key in self.model.state_dict().keys():
            num_params += self.model.state_dict()[key].numel()
        print('{} parameters: {:.2f}M'.format(self.args.model, num_params / 1e6))
        # print(*self.model.state_dict().keys())
        '''if self.args.cid == 0:
            for name, param in self.model.named_parameters():
                print(name, param.shape)'''

        if hasattr(self, 'local_keys'):
            print('client {} local_keys:'.format(self.args.cid), self.local_keys)

        if self.args.strategy in ['FedICRA']:
            self.start_phase = True

    def forward(self, x):
        return self.model(x)

    def get_weights(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights, config):
        # print('Setting weights')
        # FedAA, FedICRA
        if self.args.strategy in ['FedICRA']:
            eta = 1.0
            num_pre_loss = 10
            threshold = 0.1
            server_model = copy.deepcopy(self.model)
            server_state_dict = OrderedDict({
                k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)
            })
            self.model.load_state_dict(server_state_dict, strict=False)
            temp_model = copy.deepcopy(self.model)

            # ignore_keys = ['out_conv', 'up4', 'up3', 'up2','up1','down4','down3']
            ignore_keys = ['out_conv', 'up4', 'up3', 'up2','up1']
            # ignore_keys = ['out_conv', 'up4', 'up3']
            # ignore_keys = ['out_conv']
            local_keys = []
            for name, _ in self.model.named_parameters():
                for ignore_key in ignore_keys:
                    if ignore_key in name:
                        local_keys.append(name)
            # print(local_keys)

            params = list(self.model.parameters())
            server_params = list(server_model.parameters())
            temp_params = list(temp_model.parameters())

            if torch.sum(server_params[0] - params[0]) == 0:
                # print('skip', config)
                return

            if self.args.strategy in ['FedICRA'] and config['iter_global'] <= 50:
                print('skip', config)
                return

            # only consider higher layers
            def get_params_p(model):
                params_p = []
                for name, param in model.named_parameters():
                    if name in local_keys:
                        params_p.append(param)
                        # print(name)
                return params_p

            params_p = get_params_p(self.model)
            server_params_p = get_params_p(server_model)
            temp_params_p = get_params_p(temp_model)

            # frozen the lower layers to reduce computational cost in Pytorch
            for name, param in temp_model.named_parameters():
                if name in local_keys:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # initialize the weight to all ones in the beginning
            if not hasattr(self, 'weights'):
                self.fedaa_weights = [torch.ones_like(param.data).cuda() for param in params_p]

            # initialize the higher layers in the temp local model
            for temp_param, param, server_param, fedaa_weight in zip(temp_params_p, params_p, server_params_p,
                                                self.fedaa_weights):
                temp_param.data = param + (server_param - param) * fedaa_weight

            # used to obtain the gradient of higher layers
            # no need to use optimizer.step(), so lr=0
            # optimizer = torch.optim.SGD(temp_params_p, lr=0)
            optimizer = torch.optim.AdamW(temp_params_p,lr=0,betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-2, amsgrad=False)
            ce_loss = nn.CrossEntropyLoss(ignore_index=self.args.num_classes)

            # weight learning
            losses = []
            count = 0
            while True:
                for i_batch, sampled_batch in enumerate(self.trainloader):

                    if self.args.img_class == 'faz':
                        volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    elif self.args.img_class == 'odoc' or self.args.img_class == 'polyp':
                        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

                    with autocast(enabled=self.amp):
                        outputs = temp_model(volume_batch)[0]
                        loss = ce_loss(outputs, label_batch[:].long())

                    optimizer.zero_grad()
                    if self.amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    # update weight in this batch
                    for temp_param, param, server_param, fedaa_weight in zip(temp_params_p, params_p, server_params_p,
                                                self.fedaa_weights):
                        # print(type(temp_param), type(server_param), type(param))
                        # print(param)
                        if temp_param.grad == None: # ignore calculation when no gradient given
                            continue
                        fedaa_weight.data = torch.clamp(
                            fedaa_weight - eta * (temp_param.grad * (server_param - param)), 0, 1)

                    # update temp local model in this batch
                    for temp_param, param, server_param, fedaa_weight in zip(temp_params_p, params_p, server_params_p,
                                                self.fedaa_weights):
                        temp_param.data = param + (server_param - param) * fedaa_weight

                losses.append(loss.item())
                count += 1

                print('Client:', self.args.cid, '\tStd:', np.std(losses[-num_pre_loss:]),
                    '\tALA epochs:', count, self.start_phase)

                # only train one epoch in the subsequent iterations
                if not self.start_phase:
                    break

                # train the weight until convergence
                if len(losses) > num_pre_loss and np.std(losses[-num_pre_loss:]) < threshold:
                    print('Client:', self.args.cid, '\tStd:', np.std(losses[-num_pre_loss:]),
                        '\tALA epochs:', count)
                    break

            self.start_phase = False

            # obtain initialized local model
            for param, temp_param in zip(params_p, temp_params_p):
                param.data = temp_param.data.clone()

        # Other federagted algorithms
        else:
            state_dict_temp =  {
                k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), weights)
            }

            state_dict = OrderedDict(state_dict_temp)
            self.model.load_state_dict(state_dict, strict=False)


def tv_loss(predication):
    min_pool_x = nn.functional.max_pool2d(
        predication * -1, (3, 3), 1, 1) * -1
    contour = torch.relu(nn.functional.max_pool2d(
        min_pool_x, (3, 3), 1, 1) - min_pool_x)
    # length
    length = torch.mean(torch.abs(contour))
    return length


class TreeEnergyLoss(nn.Module):
    def __init__(self):
        super(TreeEnergyLoss, self).__init__()
        # self.configer = configer
        # if self.configer is None:
        #     print("self.configer is None")

        # self.weight = self.configer.get('tree_loss', 'params')['weight']
        # self.weight = weight
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02) ##pls see the paper for the sigma!!!!!

    def forward(self, preds, low_feats, high_feats, unlabeled_ROIs, weight):
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            # print("preds.size()", preds.size())
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            # print("low_feats.size()", low_feats.size())
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            # print("unlabeled_ROIs.size()", unlabeled_ROIs.size())
            N = unlabeled_ROIs.sum()
            # print('high_feats.size()', high_feats.size())
            # print("N", N)

        prob = torch.softmax(preds, dim=1)
        # print("prob.size()", prob.size())
        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

        # high-level MST
        if high_feats is not None:
            high_feats = F.interpolate(high_feats, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats)
            # print('tree.size()', tree.size())
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)  # [b, n, h, w]

        tree_loss = (unlabeled_ROIs * torch.abs(prob - AS)).sum()
        if N > 0:
            tree_loss /= N

        return weight * tree_loss, AS


class MScaleAddTreeEnergyLoss(nn.Module):
    def __init__(self):
        super(MScaleAddTreeEnergyLoss, self).__init__()
        # self.configer = configer
        # if self.configer is None:
        #     print("self.configer is None")

        # self.weight = self.configer.get('tree_loss', 'params')['weight']
        # self.weight = weight
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02) ##pls see the paper for the sigma!!!!!

    def forward(self, preds, low_feats, high_feats_1, high_feats_2, high_feats_3, unlabeled_ROIs, weight):
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            # print("preds.size()", preds.size())
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            # print("low_feats.size()", low_feats.size())
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            # print("unlabeled_ROIs.size()", unlabeled_ROIs.size())
            N = unlabeled_ROIs.sum()
            # print('high_feats.size()', high_feats.size())
            # print("N", N)

        prob = torch.softmax(preds, dim=1)
        # print("prob.size()", prob.size())
        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

        # high-level MST
        if high_feats_1 is not None:
            high_feats_1 = F.interpolate(high_feats_1, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_1)
            # print('tree.size()', tree.size())
            AS_1 = self.tree_filter_layers(feature_in=AS, embed_in=high_feats_1, tree=tree, low_tree=False)  # [b, n, h, w]
            
        if high_feats_2 is not None:
            high_feats_2 = F.interpolate(high_feats_2, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_2)
            # print('tree.size()', tree.size())
            AS_2 = self.tree_filter_layers(feature_in=AS, embed_in=high_feats_2, tree=tree, low_tree=False)  # [b, n, h, w]
            
        if high_feats_3 is not None:
            high_feats_3 = F.interpolate(high_feats_3, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_3)
            # print('tree.size()', tree.size())
            AS_3 = self.tree_filter_layers(feature_in=AS, embed_in=high_feats_3, tree=tree, low_tree=False)  # [b, n, h, w]

        tree_loss_1 = (unlabeled_ROIs * torch.abs(prob - AS_1)).sum()
        tree_loss_2 = (unlabeled_ROIs * torch.abs(prob - AS_2)).sum()
        tree_loss_3 = (unlabeled_ROIs * torch.abs(prob - AS_3)).sum()
        tree_loss = tree_loss_1 + tree_loss_2 + tree_loss_3
        
        if N > 0:
            tree_loss /= N

        return weight * tree_loss, AS_1, AS_2, AS_3


class MScaleRecurveTreeEnergyLoss(nn.Module):
    def __init__(self):
        super(MScaleRecurveTreeEnergyLoss, self).__init__()
        # self.configer = configer
        # if self.configer is None:
        #     print("self.configer is None")

        # self.weight = self.configer.get('tree_loss', 'params')['weight']
        # self.weight = weight
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=0.02) ##pls see the paper for the sigma!!!!!

    def forward(self, preds, low_feats, high_feats_1, high_feats_2, high_feats_3, unlabeled_ROIs, weight):
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            # print("preds.size()", preds.size())
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            # print("low_feats.size()", low_feats.size())
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            # print("unlabeled_ROIs.size()", unlabeled_ROIs.size())
            N = unlabeled_ROIs.sum()
            # print('high_feats.size()', high_feats.size())
            # print("N", N)

        prob = torch.softmax(preds, dim=1)
        # print("prob.size()", prob.size())
        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

        # high-level MST
        if high_feats_1 is not None:
            high_feats_1 = F.interpolate(high_feats_1, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_1)
            # print('tree.size()', tree.size())
            AS_1 = self.tree_filter_layers(feature_in=AS, embed_in=high_feats_1, tree=tree, low_tree=False)  # [b, n, h, w]
            
        if high_feats_2 is not None:
            high_feats_2 = F.interpolate(high_feats_2, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_2)
            # print('tree.size()', tree.size())
            AS_2 = self.tree_filter_layers(feature_in=AS_1, embed_in=high_feats_2, tree=tree, low_tree=False)  # [b, n, h, w]
            
        if high_feats_3 is not None:
            high_feats_3 = F.interpolate(high_feats_3, size=(h, w), mode='bilinear', align_corners=False)
            # print('new high_feats.size()', high_feats.size())
            tree = self.mst_layers(high_feats_3)
            # print('tree.size()', tree.size())
            AS_3 = self.tree_filter_layers(feature_in=AS_2, embed_in=high_feats_3, tree=tree, low_tree=False)  # [b, n, h, w]

        # tree_loss_1 = (unlabeled_ROIs * torch.abs(prob - AS_1)).sum()
        # tree_loss_2 = (unlabeled_ROIs * torch.abs(prob - AS_2)).sum()
        # tree_loss_3 = (unlabeled_ROIs * torch.abs(prob - AS_3)).sum()
        # tree_loss = tree_loss_1 + tree_loss_2 + tree_loss_3
        tree_loss = (unlabeled_ROIs * torch.abs(prob - AS_3)).sum()
        
        if N > 0:
            tree_loss /= N

        return weight * tree_loss, AS_1, AS_2, AS_3