#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile
# check the enviroment info
nvidia-smi
export PYTHONPATH="$PWD":$PYTHONPATH
#python3 -m pip install yacs
#python3 -m pip install torchcontrib
#python3 -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git

DATA_ROOT="$PWD/data"
DATA_DIR="${DATA_ROOT}/cityscapes"
SAVE_DIR="${DATA_ROOT}/seg_result/cityscapes/"

BACKBONE="deepbase_resnet101_dilated8"

CONFIGS="configs/cityscapes/R_101_D_8.json"

MODEL_NAME="deeplabv3p"
LOSS_TYPE="fs_ce_loss"
CHECKPOINTS_NAME="${MODEL_NAME}_"$2
LOG_FILE="./log/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="./pretrained/resnet101-imagenet.pth"
MAX_ITERS=40000
SIGMA=0.002


if [ "$1"x == "val"x ]; then
  python3 -u main.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0 --val_batch_size 1 --resume ./models/cityscapes/${CHECKPOINTS_NAME}.pth \
                       --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image --sigma ${SIGMA} \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_demo

  cd lib/metrics
  python3 -u cityscapes_evaluator.py --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_demo/label  \
                                       --gt_dir ${DATA_DIR}/val/label

else
  echo "$1"x" is invalid..."
fi
