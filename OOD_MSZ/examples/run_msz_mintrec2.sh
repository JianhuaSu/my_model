
for method in  'msz' 
do
    for text_backbone in 'bert-base-uncased' 
    do
        for ood_dataset in  'MIntRec2-OOD' 
        do
            for ood_detection_method in 'ma'  #'residual' 'vim' 'energy' 'maxlogit' 'msp'
            do
                CUDA_VISIBLE_DEVICES=1 python run.py \
                --dataset 'MIntRec2' \
                --data_path '/home/sharing/disk2/sjh/intentDataset' \
                --ood_dataset $ood_dataset \
                --logger_name ${method}_${ood_detection_method} \
                --multimodal_method $method \
                --method ${method}\
                --ood_detection_method $ood_detection_method \
                --train \
                --ood \
                --tune \
                --save_results \
                --gpu_id '0' \
                --video_feats_path 'swin_feats.pkl' \
                --audio_feats_path 'wavlm_feats.pkl' \
                --ood_video_feats_path 'swin_feats.pkl' \
                --ood_audio_feats_path 'wavlm_feats.pkl' \
                --text_backbone $text_backbone \
                --config_file_name msz_MIntRec2 \
                --output_path '/home/sharing/disk2/sjh/Outputs/OOD_MIntRec2/'${method} \
                --results_file_name 'results_msz_MIntRec2_ood.csv' 
            done
        done
    done
done

