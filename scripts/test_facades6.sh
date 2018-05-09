# models
RESULTS_DIR='./results/dows1'

G_PATH='/home/twak/code/bikegan/checkpoints_pub/dows1/dows1_bicycle_gan_08_05_2018_19/latest_net_G.pth'
E_PATH='/home/twak/code/bikegan/checkpoints_pub/dows1/dows1_bicycle_gan_08_05_2018_19/latest_net_E.pth'
#E_PATH='./pretrained_models/facades_label2image_net_E.pth'

# dataset
CLASS='dows1'
DIRECTION='BtoA' # from domain A to domain B
LOAD_SIZE=286 # scale images to this size
FINE_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image

# misc
GPU_ID=0   # gpu id
HOW_MANY=300 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images


# commandOB
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --sync \
  --dataroot /media/twak/8bc5e750-9a70-4180-8eee-ced2fbba6484/data/windows1 \
  --results_dir ${RESULTS_DIR} \
  --G_path ${G_PATH} \
  --E_path ${E_PATH} \
  --which_direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --how_many ${HOW_MANY} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip
