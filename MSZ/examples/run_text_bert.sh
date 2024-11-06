#!/usr/bin bash

for dataset in 'MIntRec2' #'MIntRec' 
do
    for seed in 1 2 3 4 5
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'text' \
        --method 'text' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --tune \
        --seed $seed \
        --gpu_id '3' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone 'bert-base-uncased' \
        --config_file_name 'text_bert_'$dataset \
        --results_file_name 'text_bert.csv' \
        --output_path '/home/sharing/disk2/sjh/Outputs/'$dataset
    done
done