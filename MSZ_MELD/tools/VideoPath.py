import csv
import os
import pickle
from tqdm import tqdm


class VideoLoadPath:

    def __init__(self, path=None):
        
        self.path = path
        
    def __call__(self, dataset, data_path):

        video_path_dict = {}
        indexs = self.get_indexes(dataset, data_path)
        
        video_path_list = self.find_video(dataset, data_path, indexs)

        for idx, index in enumerate(tqdm(indexs, desc="Iteration")):
            video_path_dict[index] = video_path_list[idx]
            print(f'label:{index},path:{video_path_list[idx]}')

        print("****路径写入.pkl文件****")
        with open('/home/sharing/disk1/zhanghanlei/Datasets/public/MIntRec/video_path/video_path.pkl', 'wb') as f:
            pickle.dump(video_path_dict, f)
        print("***写入结束***")



    def get_indexes(self, dataset, data_path):

        with open(os.path.join(data_path, dataset, 'all.tsv'), 'r') as f:

            data = csv.reader(f, delimiter="\t")
            indexes = []

            for i, line in enumerate(data):
                if i == 0:
                    continue

                if dataset == 'MIntRec':
                    index = '_'.join([str(line[0]), str(line[1]), str(line[2])])
                    
                elif dataset in ['MIntRec2.0','MELD-DA']:
                    index = '_'.join(['dia' + str(line[0]), 'utt' + str(line[1])])

                elif dataset in ['MOSI', 'MOSEI','IEMOCAP']:
                    index = str(line[0])
                    
                elif dataset in ['AnnoMi']:
                    index = '_'.join([str(line[0]), str(line[1])])

                indexes.append(index)

        return indexes



    def find_video(self, dataset, data_path, indexes):

        path = os.path.join(data_path, dataset)
        file_paths = []

        for i, name in enumerate(indexes):

            if dataset in ['MOSI', 'MOSEI']:
                split_name = name.split("$_$")
                target_path = os.path.join(path, 'raw_data', split_name[0], split_name[1] + '.mp4')

            elif dataset in ['IEMOCAP', 'MELD-DA', 'MIntRec2.0']:
                target_path = os.path.join(path, 'raw_data', name + '.mp4')

            elif dataset in ['MIntRec']:
                split_name = name.split("_")
                target_path = os.path.join(path, 'raw_data', split_name[0], split_name[1], split_name[2] + '.mp4')

            elif dataset in ['AnnoMi']:
                target_path = os.path.join(path, name + '.mp4')

            file_paths.append(target_path)

        return file_paths



