# models
RESULTS_DIR='./results/empty2windows_f005'
G_PATH='./checkpoints/empty2windows_f005/empty2windows_f005_bicycle_gan/latest_net_G.pth'
E_PATH='./checkpoints/empty2windows_f005/empty2windows_f005_bicycle_gan/latest_net_E.pth'

# dataset
DATA='cmpjohn_empty2windows'
PHASE='val'
NZ=8
DIRECTION='BtoA' # from domain A to domain B
LOAD_SIZE=286 # scale images to this size
FINE_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image

# misc
GPU_ID=0   # gpu id
HOW_MANY=20 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images

  # --test_image_z \
  # --sync

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test_interactive.py \
  --dataroot ./datasets/${DATA} \
  --phase ${PHASE} \
  --results_dir ${RESULTS_DIR} \
  --G_path ${G_PATH} \
  --E_path ${E_PATH} \
  --which_direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --nz ${NZ} \
  --how_many ${HOW_MANY} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip \
  --imgpos_condition \
  --walldist_condition \
