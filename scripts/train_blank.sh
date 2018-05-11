CLASS='blank3'  # facades, day2night, edges2shoes, edges2handbags, maps

RESULTS_DIR='./results/'$CLASS
# G_PATH='/home/twak/code/bikegan/checkpoints/'$CLASS'/'$CLASS'/latest_net_G.pth'
# E_PATH='/home/twak/code/bikegan/checkpoints/'$CLASS'/'$CLASS'/latest_net_E.pth'

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
DIRECTION='BtoA'
LOAD_SIZE=286
FINE_SIZE=256
INPUT_NC=3
NITER=200
NITER_DECAY=200

# command
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
  --use_dropout \
  --which_model_netG outernet \
  --which_model_netD big_256_multi \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --save_epoch_freq 5
  --lambda_L1 1
