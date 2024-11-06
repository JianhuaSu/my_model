#!/usr/bin bash

for dataset in 'MIntRec2' # 'MIntRec2'
do
        for seed in 4 5
        do
            python run.py \
            --dataset $dataset \
            --logger_name 'msz' \
            --method 'msz' \
            --data_mode 'multi-class' \
            --train \
            --tune \
            --save_results \
            --seed $seed \
            --gpu_id '0' \
            --video_feats_path 'video_feats.pkl' \
            --audio_feats_path 'audio_feats.pkl' \
            --text_backbone 'bert-base-uncased' \
            --config_file_name 'msz_'$dataset \
            --results_file_name 'ablation_msz.csv' \
            --output_path '/home/sharing/disk2/sjh/Outputs/ablation_'$dataset
    done
done