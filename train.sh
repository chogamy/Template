#!/bin/bash

python main.py \
    --mode train \
    --task e1c1 \
    --trainer e100b32lr5e-4d0.1 \
    --callbacks earlystopping,modelcheckpoint \
    --optimizers adamw \
    --lrscheduler constant \
    --enc monologg/koelectra-base-v3-discriminator \
    --wrapper PL \
    --data jeanlee/kmhas_korean_hate_speech \

# export TOKENIZERS_PARALLELISM=false

# # K+1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# datas=(stackoverflow banking oos) # stackoverflow banking oos
# plms=(bert-base-uncased roberta-base google/electra-base-discriminator distilbert-base-uncased)
# # plms=(bert-base-uncased)
# seeds=(0 1 2 3 4 5 6 7 8 9) # 0 1 2 3 4 5 6 7 8 9
# ratios=(0.25 0.5 0.75) # 0.25 0.5 0.75

# for data in ${datas[@]}
# do
#     for plm in ${plms[@]}
#     do
#         for ratio in ${ratios[@]}
#         do
#             for seed in ${seeds[@]}
#             do
#             # K+1 training
#             python train.py \
#             --model_name_or_path ${plm} \
#             --known_cls_ratio ${ratio} \
#             --sampler samples/sampler/Convex.yaml \
#             --model samples/model/K_1.yaml --data samples/data/${data}.yaml \
#             --trainer samples/trainer/adb.yaml \
#             --seed ${seed} --mode train
#             python train.py \
#             --config ./trainer_logs/K_1_way_${plm}_${data}${ratio}_${seed}.yaml --mode test
#             # rm -rf /root/openlib/outputs
#             # rm -rf /root/openlib/data/preprocessed
#             done
#         done
#     done
# done
