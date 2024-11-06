#!/usr/bin bash

for dataset in 'MIntRec2'  #'MIntRec2' 
do
    for seed in 4 5
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'mmim' \
        --method 'mmim' \
        --data_mode 'multi-class' \
        --train \
        --tune \
        --save_results \
        --seed $seed \
        --gpu_id '1' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone 'bert-base-uncased' \
        --config_file_name 'mmim_bert_'$dataset \
        --results_file_name 'mmim_bert_'$dataset'.csv' \
        --output_path '/home/sharing/disk3/wangyifan/sujianhua/MSZ_1/outputs/'$dataset
    done
done