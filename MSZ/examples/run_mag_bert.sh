#!/usr/bin bash

for dataset in 'MIntRec2'
do
    for seed in 5 3 2 1
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'mag_bert' \
        --method 'mag_bert' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --tune \
        --seed $seed \
        --gpu_id '2' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone 'bert-base-uncased' \
        --config_file_name 'mag_bert_'$dataset \
        --results_file_name 'mag_bert.csv' \
        --output_path '/home/sharing/disk2/sjh/Outputs/'$dataset
    done
done