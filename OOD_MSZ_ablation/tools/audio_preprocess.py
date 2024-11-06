from moviepy.editor import *
from tqdm import tqdm
from transformers import AutoProcessor, WavLMModel, HubertModel
import os
import argparse
import pickle
import argparse
import librosa
import numpy as np
import logging
import torch

__all__ = ['AudioFeature']

audio_pretrained_model_path = {
    'wav2vec2.0': 'facebook/wav2vec2-base-960h',
    'WavLM': 'patrickvonplaten/wavlm-libri-clean-100h-base-plus',
    'HuBERT': 'facebook/hubert-large-ls960-ft'
}

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_video_path', type=str, default='/home/sharing/Datasets/L-M-8081/L-MIntRec/raw_data', help="The directory of the raw video path.")
    parser.add_argument('--audio_data_path', type=str, default='/home/sharing/disk1/zhanghanlei/Datasets/public/MIntRec/audio_data', help="The directory of the audio data path.")
    parser.add_argument('--raw_audio_path', type=str, default='raw_audio', help="The directory of the raw audio path.")
    parser.add_argument("--audio_feats_path", type=str, default='audio_feats.pkl', help="The directory of audio features.")

    args = parser.parse_args()

    return args

class AudioFeature:
    
    def __init__(self, args, get_raw_audio = False):

        audio_pretrained_model = audio_pretrained_model_path[args.audio_pretrained_model]

        self.processor = AutoProcessor.from_pretrained(audio_pretrained_model)
        if args.audio_pretrained_model == 'WavLM':
            self.model = WavLMModel.from_pretrained(audio_pretrained_model)
        elif args.audio_pretrained_model == 'wav2vec2.0':
            self.model = Wav2Vec2Model.from_pretrained(audio_pretrained_model)
        elif args.audio_pretrained_model == 'HuBERT':
            self.model = HubertModel.from_pretrained(audio_pretrained_model)

        self.sr = args.sr

        if get_raw_audio:
            self.__get_raw_audio(args)
    
        audio_feats = self.__gen_feats_from_audio(args, use_pretrained_model=True)
        self.__save_audio_feats(args, audio_feats)

    def __get_raw_audio(self, args):

        raw_audio_path = os.path.join(args.audio_data_path, args.raw_audio_path)

        if not os.path.exists(raw_audio_path):
            os.makedirs(raw_audio_path)       
        
        for season in tqdm(os.listdir(args.raw_video_path), desc = 'Season'):

            episode_path = os.path.join(args.raw_video_path, season)

            for episode in tqdm(os.listdir(episode_path), desc = 'Episode'):
                
                clip_path = os.path.join(episode_path, episode)
                audio_data_path = os.path.join(raw_audio_path, season, episode)
                if not os.path.exists(audio_data_path):
                    os.makedirs(audio_data_path)

                for clip in tqdm(os.listdir(clip_path), desc = 'Clip'):
                    
                    video_path = os.path.join(clip_path, clip)
                    print(video_path)
                    
                    video_name = clip.split('.')[0]

                    if os.path.exists(os.path.join(audio_data_path, video_name + ".wav")):
                        continue

                    video_segments = VideoFileClip(video_path)
                    audio = video_segments.audio
                    audio.write_audiofile(os.path.join(audio_data_path, video_name + ".wav"))
    
    def __gen_feats_from_audio(self, args, use_pretrained_model=False):
    
        audio_feats = {}
        raw_audio_path = os.path.join(args.audio_data_path, args.raw_audio_path)

        for s_path in tqdm(os.listdir(raw_audio_path),  desc = 'Season'):
            
            # audio_id = s_path[:-4]
            # read_file_path = os.path.join(raw_audio_path, s_path)
                    
            # if use_pretrained_model:
            #     pretrained_model_feats = self.__process_audio(read_file_path)
            #     audio_feats[audio_id] = pretrained_model_feats
            # else:
            #     mfcc = self.__process_audio(read_file_path)
            #     audio_feats[audio_id] = mfcc

            s_path_dir = os.path.join(raw_audio_path, s_path)

            for e_path in tqdm(os.listdir(s_path_dir), desc = 'Episode'):
                e_path_dir = os.path.join(s_path_dir, e_path)
                
                for file in tqdm(os.listdir(e_path_dir), desc = 'Clip'):

                    audio_id = '_'.join([s_path, e_path, file[:-4]])
                    read_file_path = os.path.join(e_path_dir, file)
                    
                    if use_pretrained_model:
                        pretrained_model_feats = self.__process_audio(read_file_path)
                        audio_feats[audio_id] = pretrained_model_feats
                    else:
                        mfcc = self.__process_audio(read_file_path)
                        audio_feats[audio_id] = mfcc

        return audio_feats

    def __process_audio(self, read_file_path):
        
        print('zzzzz', read_file_path)

        y, sr = librosa.load(read_file_path, sr = self.sr)
        audio_feats = self.processor(y, sampling_rate = 16000, return_tensors="pt").input_values
        with torch.no_grad():
            audio_feats = self.model(audio_feats).last_hidden_state.squeeze(0)
        
        return audio_feats

    def __save_audio_feats(self, args, audio_feats):

        audio_feats_path = os.path.join(args.audio_data_path, args.audio_feats_path)

        with open(audio_feats_path, 'wb') as f:
            pickle.dump(audio_feats, f)

if __name__ == '__main__':
    
    args = parse_arguments()
    # dataset = 'TED-OOD'
    # args.raw_video_path = '/home/sharing/disk1/zhanghanlei/Datasets/public/TED-OOD/raw_data_add'
    # args.audio_data_path = '/home/sharing/disk1/zhanghanlei/Datasets/public/TED-OOD/audio_data_add'
    # args.raw_audio_path = '/home/sharing/disk1/zhanghanlei/Datasets/public/TED-OOD/audio_data_add/raw_audio'
    # args.audio_feats_path = 'ood_audio_feats_add.pkl'
    # args.sr = 8000 if dataset == 'TED-OOD' else 16000

    args.raw_video_path = '/home/sharing/disk1/zhanghanlei/Datasets/public/MIntRec/raw_data'
    args.audio_data_path = '/home/sharing/disk1/zhanghanlei/Datasets/public/MIntRec/audio_data'
    args.raw_audio_path = '/home/sharing/disk1/zhanghanlei/Datasets/public/MIntRec/audio_data/raw_audio'
    args.audio_feats_path = 'wavlm_feats.pkl'
    args.audio_pretrained_model = 'WavLM'
    
    args.sr = 16000
    
    audio_data = AudioFeature(args, get_raw_audio = False)