#!/bin/bash

BS=64
GPU_ID=0
MAX_GITER=50000
DATA_PATH=./data
DATASET=mnist
DATAROOT=${DATA_PATH}/mnist
ISIZE=32
NC=1
NOISE_DIM=10
MODEL=cfgangp
DOUT_DIM=${NOISE_DIM}
NUM_FREQS=8
WEIGHT=gaussian_ecfd
SIGMA=0.

cmd="python src/main.py\
 --dataset ${DATASET}\
 --dataroot ${DATAROOT}\
 --model ${MODEL}\
 --batch_size ${BS}\
 --image_size ${ISIZE}\
 --nc ${NC}\
 --noise_dim ${NOISE_DIM}\
 --dout_dim ${DOUT_DIM}\
 --max_giter ${MAX_GITER}\
 --resultsroot ./out
 --gpu_device ${GPU_ID}"

if [ ${MODEL} == 'cfgangp' ]; then
    cmd+=" --num_freqs ${NUM_FREQS} --weight ${WEIGHT} --sigmas ${SIGMA}"
fi

echo $cmd
eval $cmd
