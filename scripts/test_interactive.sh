NZ=8 # style latent size
DIRECTION='BtoA' # from domain A to domain B
LOAD_SIZE=286 # scale images to this size
FINE_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image
GPU_ID=0   # gpu id

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test_interactive.py \
  --dataroot './input' \
  --which_direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --nz ${NZ} \
  --no_flip \
  --nThreads 0 \
