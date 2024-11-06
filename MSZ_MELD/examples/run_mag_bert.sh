#!/usr/bin bash

for dataset in 'MELD-DA'
do
    for seed in 2 3 4 5
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
        --gpu_id '0' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone 'bert-base-uncased' \
        --config_file_name 'mag_bert_'$dataset \
        --results_file_name 'mag_bert_'$dataset'.csv' \
        --output_path '/home/sharing/disk3/wangyifan/sujianhua/MSZ/outputs/'$dataset
    done
done