
for method in  'msz' 
do
    for text_backbone in 'bert-base-uncased' 
    do
        for ood_dataset in  'MIntRec2-OOD' 
        do
            for ood_detection_method in  'residual' 'vim' 'energy' 'maxlogit' 'msp'
            do
                CUDA_VISIBLE_DEVICES=2 python run.py \
                --dataset 'MIntRec2' \
                --data_path '/home/sharing/disk2/sjh/intentDataset' \
                --ood_dataset $ood_dataset \
                --logger_name ${method}_${ood_detection_method} \
                --multimodal_method $method \
                --method ${method}\
                --ood_detection_method $ood_detection_method \
                --ood \
                --tune \
                --save_results \
                --gpu_id '2' \
                --video_feats_path 'swin_feats.pkl' \
                --audio_feats_path 'wavlm_feats.pkl' \
                --ood_video_feats_path 'swin_feats.pkl' \
                --ood_audio_feats_path 'wavlm_feats.pkl' \
                --text_backbone $text_backbone \
                --config_file_name msz_MIntRec22 \
                --output_path '/home/sharing/disk2/sjh/Outputs/OOD_MIntRec2/ablation/no_CosClass_'${method} \
                --results_file_name 'results_msz_ablation_MIntRec2.csv' 
            done
        done
    done
done

