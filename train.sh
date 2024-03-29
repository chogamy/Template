#!/bin/bash

export TOKENIZERS_PARALLELISM=false

python main.py \
    --mode train \
    --model e1c1.E1C1 \
    --max_length 128 \
    --trainer e100b32lr5e-4 \
    --callbacks earlystopping,modelcheckpoint \
    --optimizers adamw \
    --lrscheduler constant \
    --enc monologg/koelectra-base-v3-discriminator \
    --wrapper lightning.pytorch \
    --data jeanlee/kmhas_korean_hate_speech \
    --model_path ./save_dir \


