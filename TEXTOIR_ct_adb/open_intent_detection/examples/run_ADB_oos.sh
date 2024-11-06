#!/usr/bin bash

for dataset in 'oos'
do
    for known_cls_ratio in 0.75
    do
        for labeled_ratio in 1.0
        do
            for seed in 1 2 3 4 5
            do 
                python run.py \
                --dataset $dataset \
                --method 'ADB' \
                --known_cls_ratio $known_cls_ratio \
                --labeled_ratio $labeled_ratio \
                --seed $seed \
                --backbone 'bert' \
                --config_file_name 'ADB_oos' \
                --loss_fct 'CrossEntropyLoss' \
                --gpu_id '1' \
                --train \
                --tune \
                --pretrain \
                --results_file_name 'results_ADB_oos.csv' \
                --save_results \
                --save_model
            done
        done
    done
done
