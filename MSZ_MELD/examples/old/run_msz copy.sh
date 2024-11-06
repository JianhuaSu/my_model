#!/usr/bin bash

for dataset in 'MIntRec'
do
    for seed in 1 2 3 
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'msz' \
        --method 'msz' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '1' \
        --video_feats_path 'swin_feats.pkl' \
        --audio_feats_path 'wavlm_feats.pkl' \
        --text_backbone 'bert-base-uncased' \
        --config_file_name 'msz' \
        --results_file_name 'msz.csv'
    done
done