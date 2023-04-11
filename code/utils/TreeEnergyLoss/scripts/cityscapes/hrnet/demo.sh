#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../
. config.profile
# check the enviroment info
nvidia-smi
#python3 -m pip install yacs
#python3 -m pip install torchcontrib
#python3 -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git

export PYTHONPATH="$PWD":$PYTHONPATH

DATA_ROOT="$PWD/data"
DATA_DIR="${DATA_ROOT}/cityscapes"
SAVE_DIR="${DATA_ROOT}/seg_result/cityscapes/"
BACKBONE="hrnet48"

CONFIGS="configs/cityscapes/H_48_D_4.json"

MODEL_NAME="hrnet_w48"
CHECKPOINTS_NAME="${MODEL_NAME}_"$2

LOSS_TYPE="fs_ce_loss"
LOG_FILE="./log/cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="./pretrained/hrnetv2_w48_imagenet_pretrained.pth"
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