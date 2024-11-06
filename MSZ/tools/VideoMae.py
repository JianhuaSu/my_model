import av
import numpy as np
from transformers import AutoImageProcessor, VideoMAEModel
from torch import nn



class VideoMae(nn.Module):

    def __init__(self):
        
        super(VideoMae, self).__init__()

        self.model = VideoMAEModel.from_pretrained('/home/zhanghanlei/sujianhua/MIntRec_videomae/videomae-base')
        
    def forward(self, input_list):

        outputs = self.model(**input_list)
        last_hidden_state = outputs.last_hidden_state
        
        return last_hidden_state




class VideoImag:

    def __init__(self, args):
                
        self.image_processor = AutoImageProcessor.from_pretrained('/home/zhanghanlei/sujianhua/MIntRec_videomae/videomae-base')
        self.args = args

    def __call__(self, video_paths):
        
        video_image = self.read_and_sample_videos(video_paths, clip_len=16, frame_sample_rate=2)
        input_list = self.image_processor(video_image, return_tensors="pt")
        
        return input_list.to(self.args.device)
        
    
    def read_and_sample_videos(self, video_list, clip_len, frame_sample_rate):
        '''
        Decode a batch of videos with PyAV decoder and sample frame indices.
        Args:
            video_list (`List[str]`): List of paths to video files.
            clip_len (`List[int]`): List of total number of frames to sample for each video.
            frame_sample_rate (`List[int]`): List of sampling rates for each video.
        Returns:
            result (List[List[np.ndarray]]): List of lists of np arrays of decoded frames for each video of shape (num_frames, height, width, 3).
        '''            
        results = [[] for _ in range(len(video_list))]
        for i, video_path in enumerate(video_list):
            
            container = av.open(video_path)
            num_frames = container.streams.video[0].frames
            seg_len = num_frames

            converted_len = int(clip_len * frame_sample_rate)
            
            if converted_len >= seg_len:
                # Handle the case when converted_len is greater than or equal to seg_len
                # Set converted_len to a value smaller than seg_len
                converted_len = seg_len - 1
                
            start_idx = np.random.randint(0, seg_len - converted_len)
            end_idx = start_idx + converted_len
            sampled_indices = np.linspace(start_idx, end_idx, num=clip_len)
            sampled_indices = np.clip(sampled_indices, 0, seg_len - 1).astype(np.int64)
            frames = []
            container.seek(0)
            for j, frame in enumerate(container.decode(video=0)):
                if j > end_idx:
                    break
                if j >= start_idx and j in sampled_indices:
                    frames.append(frame)
            
            results[i] = [x.to_ndarray(format="rgb24") for x in frames]
            
            if len(results[i]) < clip_len:
                n = clip_len - len(results[i])
                for _ in range(n):
                    array = np.zeros(results[i][0].shape)
                    results[i].append(array)
         
        return results