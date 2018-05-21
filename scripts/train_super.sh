CLASS='super9'
MODEL='bicycle_gan'
GPU_ID=0
DISPLAY_ID=$((GPU_ID*10+1))
PORT=8097
NZ=8

CHECKPOINTS_DIR=./checkpoints_pub/${CLASS}/
DATE=`date '+%d_%m_%Y_%H'`
NAME=${CLASS} #_${MODEL}_${DATE}


# dataset
NO_FLIP=''
DIRECTION='AtoB'
LOAD_SIZE=286
FINE_SIZE=256
INPUT_NC=3

NITER=200
NITER_DECAY=200
SAVE_EPOCH=25
DIRECTION='BtoA'

#  --dataroot /media/twak/8bc5e750-9a70-4180-8eee-ced2fbba6484/data/regent_style_8k \
# command
#  --continue_train \

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot /data \
  --name ${NAME} \
  --model ${MODEL} \
  --display_port ${PORT} \
  --which_direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --nz ${NZ} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --use_dropout \
  --conditional_D \
  --dataset_mode blur

