#!/usr/bin bash

for dataset in 'MIntRec2' 
do
    for seed in 1 3 4 5
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'mult' \
        --method 'mult' \
        --data_mode 'multi-class' \
        --train \
        --tune \
        --save_results \
        --seed $seed \
        --gpu_id '3' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone 'bert-base-uncased' \
        --config_file_name 'mult_bert_'$dataset \
        --results_file_name 'mult_bert.csv' \
        --output_path '/home/sharing/disk2/sjh/Outputs/'$dataset
    done
done