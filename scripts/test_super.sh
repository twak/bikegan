CLASS='super3'
MODEL='bicycle_gan'


RESULTS_DIR='./results/'$CLASS
G_PATH='/home/twak/code/bikegan/checkpoints_pub/'$CLASS'/'$CLASS'/latest_net_G.pth'
E_PATH='/home/twak/code/bikegan/checkpoints_pub/'$CLASS'/'$CLASS'/latest_net_E.pth'

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
LOAD_SIZE=400
FINE_SIZE=256
INPUT_NC=3

NITER=300
NITER_DECAY=300
SAVE_EPOCH=25
DIRECTION='BtoA'

HOW_MANY=300 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --sync \
  --dataroot /media/twak/8bc5e750-9a70-4180-8eee-ced2fbba6484/data/regent_style_8k \
  --results_dir ${RESULTS_DIR} \
  --G_path ${G_PATH} \
  --E_path ${E_PATH} \
  --which_direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --how_many ${HOW_MANY} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip \
  --dataset_mode blur
