from VideoPath import VideoLoadPath

data_path = '/home/sharing/disk1/zhanghanlei/Datasets/public'
dataset = 'MIntRec'
v = VideoLoadPath()
v(dataset, data_path)

print("结束")


# from VideoPath import VideoLoadPath

# data_path = '/home/sharing/Datasets'
# dataset = 'MIntRec'
# v = VideoLoadPath(path=1)
# v(dataset, data_path)

# print("结束")

# import pickle

# with open('/home/sharing/disk1/zhanghanlei/Datasets/public/MIntRec/test/video_path.pkl', 'rb') as f:
#     data = pickle.load(f)
    
# with open('/home/sharing/disk1/zhanghanlei/Datasets/public/MIntRec/test/video_path1.txt', 'w') as f:
#     for key, value in data.items():
#         f.write(f'{key}: {value}\n')