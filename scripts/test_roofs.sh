CLASS='roofs2'  # facades, day2night, edges2shoes, edges2handbags, maps

RESULTS_DIR='./results/'$CLASS
G_PATH='/home/twak/code/bikegan/checkpoints_pub/'$CLASS'/'$CLASS'/latest_net_G.pth'
E_PATH='/home/twak/code/bikegan/checkpoints_pub/'$CLASS'/'$CLASS'/latest_net_E.pth'

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
LOAD_SIZE=512
FINE_SIZE=512
INPUT_NC=3

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --display_id ${DISPLAY_ID} \
  --dataroot /media/twak/8bc5e750-9a70-4180-8eee-ced2fbba6484/data/roofs2 \
  --results_dir ${RESULTS_DIR} \
  --name ${NAME} \
  --how_many 300 \
  --model ${MODEL} \
  --G_path ${G_PATH} \
  --E_path ${E_PATH} \
  --display_port ${PORT} \
  --which_direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --loadSize ${LOAD_SIZE} \
  --fineSize ${FINE_SIZE} \
  --nz ${NZ} \
  --which_model_netE 'resnet_512' \
  --input_nc ${INPUT_NC} \
  --no_flip
