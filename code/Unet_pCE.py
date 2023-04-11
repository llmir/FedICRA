import argparse
import logging
import os
import random
import shutil
import sys
import time
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/FAZ_h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='odoc_bs_12/pCE_Unet', help='experiment_name')
parser.add_argument('--client', type=str,
                    default='client1', help='domain NUM')
parser.add_argument('--sup_type', type=str,
                    default='mask', help='supervision label type(scr ; label ; scr_n ; keypoint ; block)')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--in_chns', type=int, default=3,
                    help='image channel')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size',nargs='+', type=int, default=[256,256],
                    help='patch size of network input')
parser.add_argument('--gpus', type=int, default=0,
                    help='gpu index,must set CUDA_VISIBLE_DEVICES at terminal')
parser.add_argument('--img_class', type=str,
                    default='faz', help='the img class(odoc or faz)')
parser.add_argument('--amp', type=bool,  default=0,
                        help='whether use amp training')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
args = parser.parse_args()

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    amp = args.amp
    if args.amp:
            scaler = GradScaler()
    model = net_factory(net_type=args.model, in_chns=args.in_chns, class_num=num_classes)
    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size,img_class=args.img_class)
    ]), client=args.client, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path,
                          client=args.client, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    dice_loss = losses.pDLoss(num_classes,ignore_index=num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            if args.img_class == 'faz':
                volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            elif args.img_class == 'odoc':
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
    
            with autocast(enabled=amp):
                out = model(volume_batch)
                if args.model == 'fcnet':
                    high_feats, outputs = out
                elif args.model in ['deeplabv3plus', 'treefcn']:
                    outputs, _, high_feats = out
                elif args.model == 'unet_head':
                    outputs, feature, de1, de2, de3, de4, high_feats1, high_feats2 = out
                else:
                    outputs, feature, de1, de2, de3, de4 = out
                    high_feats = de2
                outputs_soft = torch.softmax(outputs, dim=1)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss = loss_ce
            optimizer.zero_grad()
            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f' %
                (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:, :, :]
                
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs2 = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs2[0,0,...] * 50, iter_num,dataformats='HW')

                labs = label_batch[0, ...].unsqueeze(0) * 50

                writer.add_image('train/GroundTruth', labs, iter_num,dataformats='CHW')

            if iter_num > 0 and iter_num % 20 == 0:
                model.eval()
                metric_list = 0.0
                loss_val = 0.0
    
                for i_batch, sampled_batch in enumerate(valloader):
                    if args.img_class == 'faz':
                        volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    elif args.img_class == 'odoc':
                        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    with torch.no_grad():
                        outputs_val = model(volume_batch)[0]
                        outputs_soft_val = torch.softmax(outputs_val, dim=1)
                        loss_ce_val = ce_loss(outputs_val, label_batch[:].long())
                        loss = 0.5 * (loss_ce_val + dice_loss(outputs_soft_val,
                            label_batch.unsqueeze(1)))
                        loss_val+=loss
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list = metric_list+np.array(metric_i)
                    
                loss_val=loss_val/len(db_val)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_recall'.format(class_i+1),
                                      metric_list[class_i, 2], iter_num)
                    writer.add_scalar('info/val_{}_precision'.format(class_i+1),
                                      metric_list[class_i, 3], iter_num)
                    writer.add_scalar('info/val_{}_jc'.format(class_i+1),
                                      metric_list[class_i, 4], iter_num)
                    writer.add_scalar('info/val_{}_specificity'.format(class_i+1),
                                      metric_list[class_i, 5], iter_num)
                    writer.add_scalar('info/val_{}_ravd'.format(class_i+1),
                                      metric_list[class_i, 6], iter_num)
                    writer.add_scalar('info/total_loss_val', loss_val, iter_num)
    

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                mean_recall = np.mean(metric_list, axis=0)[2]
                mean_precision = np.mean(metric_list, axis=0)[3]
                mean_jc = np.mean(metric_list, axis=0)[4]
                mean_specificity = np.mean(metric_list, axis=0)[5]
                mean_ravd = np.mean(metric_list, axis=0)[6]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)
                writer.add_scalar('info/val_mean_recall', mean_recall, iter_num)
                writer.add_scalar('info/val_mean_precision', mean_precision, iter_num)
                writer.add_scalar('info/val_mean_jc', mean_jc, iter_num)
                writer.add_scalar('info/val_mean_specificity', mean_specificity, iter_num)
                writer.add_scalar('info/val_mean_ravd', mean_ravd, iter_num)
                

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f : mean_recall : %f mean_precision : %f : mean_jc : %f mean_specificity : %f : mean_ravd : %f : total_loss : %f ' % (iter_num, performance, mean_hd95, mean_recall, mean_precision, mean_jc, mean_specificity, mean_ravd,loss_val))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.client, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
