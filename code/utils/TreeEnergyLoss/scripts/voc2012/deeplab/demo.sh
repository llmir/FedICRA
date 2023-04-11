#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile
# check the enviroment info
nvidia-smi

export PYTHONPATH="$PWD":$PYTHONPATH
DATA_ROOT="$PWD/data"
DATA_DIR="${DATA_ROOT}/voc2012"
SAVE_DIR="${DATA_ROOT}/seg_result/voc2012/"

BACKBONE="deepbase_resnet101_dilated16"

CONFIGS="configs/voc2012/R_101_D_16.json"

MODEL_NAME="deeplabv3p"
LOSS_TYPE="fs_ce_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_"$2
LOG_FILE="./log/voc2012/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="./pretrained/resnet101-imagenet.pth"

MAX_ITERS=80000
SIGMA=0.02

if [ "$1"x == "val"x ]; then
  python3 -u main.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 --val_batch_size 1 \
                       --resume ./models/voc2012/${CHECKPOINTS_NAME}.pth \
                       --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image --sigma ${SIGMA} \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_demo

  cd lib/metrics
  python3 -u pascal_context_evaluator.py --configs /data/liangzhiyuan/projects/openseg2/configs/voc2012/R_101_D_16.json \
                                         --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_demo/label \
                                         --gt_dir ${DATA_DIR}/val/label

else
  echo "$1"x" is invalid..."
fi