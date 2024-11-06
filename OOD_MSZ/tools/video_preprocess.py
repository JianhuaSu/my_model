import cv2
import os
import torch
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import copy
import json
import pickle
import argparse

from torch import nn
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.core import encode_mask_results
from mmdet.core.visualization import imshow_det_bboxes
from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
from mmcv.runner import load_checkpoint

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

EPS = 1e-2

def parse_arguments():
    
    # CUDA_VISIBLE_DEVICES=0 
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_checkpoint_path', type=str, default='/home/sharing/disk1/zhanghanlei/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', help="The directory of the detection checkpoint path.")
    parser.add_argument('--detection_config_path', type=str, default='/home/sharing/disk1/zhanghanlei/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', help="The directory of the detection configuration path.")
    parser.add_argument('--video_data_path', type=str, default='/home/sharing/disk1/zhanghanlei/Datasets/public/MIntRec/video_data', help="The directory of the video data path.")
    parser.add_argument('--video_feats_path', type=str, default='video_feats.pkl', help="The directory of the video features path.")
    parser.add_argument('--raw_video_path', type=str, default='/home/sharing/disk1/zhanghanlei/Datasets/private/tests/S05/E02', help="The directory of the raw video segments.")
    parser.add_argument('--frames_path', type=str, default='/home/sharing/disk1/zhanghanlei/Datasets/public/MIntRec/screenshots', help="The directory of frames with bbox.")
    parser.add_argument('--frames_bbox_path', type=str, default='add_screenshots_bbox', help="The directory of frames with bbox.")
    parser.add_argument('--frame_frequency', type=int, default=10, help="The frequency of extracting frames.")
    parser.add_argument('--speaker_annotation_path', type=str, default='/home/sharing/disk1/zhanghanlei/Datasets/public/MIntRec/video_data/speaker_annotations.json', help="The original file of annotated speaker ids.")
    parser.add_argument('--TalkNet_speaker_path', type=str, default='/home/sharing/disk1/zhanghanlei/Datasets/private/tests', help="The output directory of TalkNet model.")
    parser.add_argument("--use_TalkNet", action="store_true", help="whether using the annotations from TalkNet to get video features.")
    parser.add_argument("--roi_feat_size", type = int, default=7, help="The size of Faster R-CNN region of interest.")

    args = parser.parse_args()

    return args

def process_raw_video(read_file_path, output_file_path, output_file_name, frame_frequency):

    cap = cv2.VideoCapture(read_file_path)

    if not cap.isOpened():
        print('The directory is wrong.')
    
    cnt = 0

    while True:

        ret, frame = cap.read()
        if frame is None:
            break
        
        if cnt % frame_frequency == 0:
            write_path = os.path.join(output_file_path, output_file_name + '_' + str(cnt) + '.jpg')
            cv2.imwrite(write_path, frame)
        cnt += 1

        if not ret:
            break

def generate_screenshots(args, sample_ids = None):

    '''
    Input:
        args.video_data_path
        args.frames_path
        args.raw_video_path
        
    '''
    print('Start generating screenshots...')

    frames_path = os.path.join(args.video_data_path, args.frames_path)

    if not os.path.exists(frames_path):
        os.makedirs(frames_path)

    for s_path in tqdm(os.listdir(args.raw_video_path), desc = 'Season'):
    
        s_path_dir = os.path.join(args.raw_video_path, s_path)

        for e_path in tqdm(os.listdir(s_path_dir), desc = 'Episode'):
            e_path_dir = os.path.join(s_path_dir, e_path)

            for file in tqdm(os.listdir(e_path_dir), desc = 'Clip'):
                output_file_name = str(s_path) + '_' + str(e_path) + '_' + str(file)
                if sample_ids is not None:
                    print(output_file_name[:-4])
                    if output_file_name[:-4] not in sample_ids:
                        continue
                        
                read_file_path = os.path.join(e_path_dir, file)

                process_raw_video(read_file_path, frames_path, output_file_name[:-4], args.frame_frequency)

def color_val_matplotlib(color):
    """Convert various input in BGR order to normalized RGB matplotlib color
    tuples,

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    color = mmcv.color_val(color)
    color = [color / 255 for color in color[::-1]]
    return tuple(color)

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      segms=None,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      mask_color=None,
                      thickness=2,
                      font_size=100,
                      win_name='',
                      show=True,
                      get_person_id = False,
                      wait_time=0,
                      out_file=None):

    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray or None): Masks, shaped (n,h,w) or None
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.  Default: 0
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (str or tuple(int) or :obj:`Color`, optional):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        show (bool): Whether to show the image. Default: True
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes.shape[0] == labels.shape[0], \
        'bboxes.shape[0] and labels.shape[0] should have the same length.'
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    mask_colors = []
    if labels.shape[0] > 0:
        if mask_color is None:
            # random color
            np.random.seed(42)
            mask_colors = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
        else:
            # specify  color
            mask_colors = [
                np.array(mmcv.color_val(mask_color)[::-1], dtype=np.uint8)
            ] * (
                max(labels) + 1)

    bbox_color = color_val_matplotlib(bbox_color)
    text_color = color_val_matplotlib(text_color)

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    polygons = []
    color = []
    
    person_id = 0

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        
        if get_person_id:
            label_text = 'Id: ' + str(person_id)
            person_id += 1

        if len(bbox) > 4:
            label_text += f' | {bbox[-1]:.02f}'
        ax.text(
            bbox_int[0],
            bbox_int[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size,
            verticalalignment='top',
            horizontalalignment='left')
            
        if segms is not None:
            color_mask = mask_colors[labels[i]]
            mask = segms[i].astype(bool)
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    plt.imshow(img)

    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)

    if out_file is not None:

        if get_person_id:
            out_file = out_file.strip('.jpg') + '_' + str(person_id) + '.jpg'

        mmcv.imwrite(img, out_file)

    plt.close()

    return img

def predict_bbox(model, imgs):

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)


    if not is_batch:
        return results[0]
    else:
        return results

def draw_bbox(img,
                    result,
                    score_thr=0.8,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    get_person_id = False,
                    out_file=None,
                    classes = None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=classes,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            get_person_id=get_person_id,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

def generate_frames_bbox(args):
    '''
    Input:
        args.frames_path
        args.frames_bbox_path

    '''
    print('Start generating bbox of frames...')

    model = init_detector(args.detection_config_path, args.detection_checkpoint_path, device='cuda:0')
    checkpoint = load_checkpoint(model, args.detection_checkpoint_path)
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        classes = checkpoint['meta']['CLASSES']
        map_class = {i:v for i, v in enumerate(classes)}

    print('COCO dataset class mapping is:', map_class)
    
    if not os.path.exists(args.frames_path):
        raise Exception('Error: The input path is empty.')
    
    frames_bbox_path = os.path.join(args.video_data_path, args.frames_bbox_path)
    if not os.path.exists(frames_bbox_path):
        os.makedirs(frames_bbox_path)

    for file_name in tqdm(os.listdir(args.frames_path), desc='Frames'):

        input_img_path = os.path.join(args.frames_path, file_name)

        result = predict_bbox(model, input_img_path)
        tmp_result = []

        output_img_path = os.path.join(frames_bbox_path, file_name)
        for idx, elem in enumerate(result):
            label = map_class[idx]
            if label == 'person':
                tmp_result.append(elem)
            else:
                tmp_result.append(np.ndarray(shape = (0, 5), dtype = np.float32))

        draw_bbox(input_img_path, tmp_result, get_person_id = True, out_file = output_img_path, classes = classes)

def isDigit(x):
    try:
        x = int(x)
        return isinstance(x, int)
    except ValueError:
        return False

def generate_speaker_bbox(args):

    '''
    Input:
        args.frames_path
        args.speaker_annotation_path
    Output:
        args.speaker_annotation_path
    '''

    print('Start generating bbox of speakers...')

    model = init_detector(args.detection_config_path, args.detection_checkpoint_path, device='cuda:0')
    checkpoint = load_checkpoint(model, args.detection_checkpoint_path)
    
    if 'CLASSES' in checkpoint.get('meta', {}):
        classes = checkpoint['meta']['CLASSES']
        map_class = {i:v for i, v in enumerate(classes)}

    print('COCO dataset class mapping is:', map_class)
    
    if not os.path.exists(args.speaker_annotation_path):
        raise Exception('Error: The input speaker id path is empty.')
    
    if not os.path.exists(args.frames_path):
        raise Exception('Error: The input frames path is empty.')

    speaker_annotations = {}
    with open(args.speaker_annotation_path, 'r') as f:
        speaker_annotations = json.load(f)

    error_cnt, lack_cnt = 0, 0
    frames_bbox_path = os.path.join(args.video_data_path, args.frames_bbox_path)

    for key in tqdm(speaker_annotations.keys(), desc = 'Extracting speaker bbox'):
        
        person_id = speaker_annotations[key]['id']

        if isDigit(person_id) == False:
            error_cnt += 1
            continue   
            
        person_id = int(person_id)
        frame = '_'.join(key.strip('.jpg').split('_')[:-1]) + '.jpg'
        input_img_path = os.path.join(args.frames_path, frame)

        result = predict_bbox(model, input_img_path)
        tmp_result = []

        roi = {}
        for idx, elem in enumerate(result):
            label = map_class[idx]
            
            if label == 'person':
                    
                null_elem = np.ndarray(shape = (5,), dtype = np.float32)
                tmp_elem = copy.copy(elem)

                for i, e in enumerate(tmp_elem):
                    
                    if i == person_id:
                        tmp_elem[i] = elem[person_id]
                        roi['id'] = person_id
                        roi['bbox'] = list([float(x) for x in tmp_elem[i]])

                    else:
                        tmp_elem[i] = null_elem
                    
                tmp_result.append(tmp_elem)

            else:
                tmp_result.append(np.ndarray(shape = (0, 5), dtype = np.float32))
            
            speaker_annotations[key] = roi

    print('The number of lacked annotations is {}'.format(lack_cnt))
    print('The number of error annotations is {}'.format(error_cnt))

    with open(args.speaker_annotation_path, 'w') as f:
        json.dump(speaker_annotations, f, indent=4)

class VideoFeature:
    
    def __init__(self, args):
        
        self.model, self.device = self.__init_detection_model(args)
        self.avg_pool = nn.AvgPool2d(args.roi_feat_size)

        if args.use_TalkNet:

            args.frame_frequency = 1
            self.bbox_feats = self.__get_TalkNet_features(args)
        
        else:
            self.bbox_feats = self.__get_Annotated_features(args)
        
        self.__save_video_feats(args, self.bbox_feats)

    def __init_detection_model(self, args):

        model = init_detector(args.detection_config_path, args.detection_checkpoint_path, device='cuda:0')
        device = next(model.parameters()).device  # model device
        return model, device

    def __get_TalkNet_features(self, args):
        
        '''
        Input: 
            args.TalkNet_speaker_path
            args.frame_frequency

        Output:
        The format of video features
        {
            'video_clip_id_a':[frame_a_feat, frame_b_feat, ..., frame_N_feat],
            'video_clip_id_b':[xxx]
        }
        Issues: (not qualified in annotated samples)
        S05_E21_43
        S05_E21_357
        S05_E21_110
        S05_E21_40
        S05_E16_433
        S06_E01_134
        '''

        video_feats = {}
        error_cnt = 0
        error_path = 0
        
        if not os.path.exists(args.TalkNet_speaker_path):
            os.makedirs(args.TalkNet_speaker_path)
            
        for video_clip_name in tqdm(os.listdir(args.TalkNet_speaker_path), desc = 'Video'):
            frames_path = os.path.join(args.TalkNet_speaker_path, video_clip_name, 'pyframes')
            bestperson_path = os.path.join(args.TalkNet_speaker_path, video_clip_name, 'pywork', 'best_persons.npy')
            
            if not os.path.exists(bestperson_path):
                error_path += 1
                continue

            bestpersons = np.load(bestperson_path)
            # print(bestpersons)

            for frame, bbox in tqdm(enumerate(bestpersons), desc = 'Frame'):
                if frame % args.frame_frequency != 0:
                    continue
                if (bbox[0] == 0) and (bbox[1] == 0) and (bbox[2] == 0) and (bbox[3] == 0):
                    error_cnt += 1
                    continue
                    
                frame_name = str('%06d' % frame)
                frame_path = os.path.join(frames_path, frame_name + '.jpg')

                """ 
                img = cv2.imread(img_ath)
                height, width, channel = img.shape
                roi = [0, 0, width, height] 
                """

                roi = bbox.tolist()              
                roi.insert(0, 0.)

                bbox_feat = self.__extract_roi_feats(self.model, self.device, frame_path, roi)
                bbox_feat = self.__average_pooling(bbox_feat)
                bbox_feat = bbox_feat.detach().cpu().numpy()
                
                if video_clip_name not in video_feats.keys():
                    video_feats[video_clip_name] = [bbox_feat]
                
                else:
                    video_feats[video_clip_name].append(bbox_feat)

        print('The number of error annotations is {}'.format(error_cnt))
        print('The number of error paths is {}'.format(error_path))
        
        return video_feats
            
    def __get_Annotated_features(self, args):

        '''
        Input: 
            args.video_data_path 
            args.speaker_annotation_path
            args.frames_path

        Output:
        The format of video features
        {
            'video_clip_id_a':
            {
                'length': N,
                'feats': [frame_a_feat, frame_b_feat, ..., frame_N_feat]
            },
            'video_clip_id_b':
            {
                xxx,
            }
        }
        '''

        speaker_annotation_path = os.path.join(args.video_data_path, args.speaker_annotation_path)
        speaker_annotations = json.load(open(speaker_annotation_path, 'r'))

        video_feats = {}
        error_cnt = 0

        try:
            for key in tqdm(speaker_annotations.keys(), desc = 'Frame'):
                
                if 'bbox' not in speaker_annotations[key].keys():
                    error_cnt += 1
                    continue
                
                roi = speaker_annotations[key]['bbox'][:4]
                roi.insert(0, 0.)

                frame_name = '_'.join(key.strip('.jpg').split('_')[:-1])
                frame_path = os.path.join(args.frames_path, frame_name + '.jpg')
                
                bbox_feat = self.__extract_roi_feats(self.model, self.device, frame_path, roi)
                bbox_feat = self.__average_pooling(bbox_feat)
                bbox_feat = bbox_feat.detach().cpu().numpy()
                
                video_clip_name = '_'.join(key.strip('.jpg').split('_')[:-2])

                if video_clip_name not in video_feats.keys():
                    video_feats[video_clip_name] = [bbox_feat]
                
                else:
                    video_feats[video_clip_name].append(bbox_feat)

        except Exception as e:
                print(e)

        print('The number of error annotations is {}'.format(error_cnt))

        return video_feats         

    def __save_video_feats(self, args, bbox_feats):
        
        if not os.path.exists(args.video_data_path):
            os.makedirs(args.video_data_path)
        video_feats_path = os.path.join(args.video_data_path, args.video_feats_path)

        with open(video_feats_path, 'wb') as f:
            pickle.dump(bbox_feats, f)

    def __extract_roi_feats(self, model, device, file_path, roi):
        
        roi = torch.tensor([roi]).to(device)
        cfg = model.cfg
        # prepare data
        data = dict(img_info=dict(filename = file_path), img_prefix=None)
        # build the data pipeline
        test_pipeline = Compose(cfg.data.test.pipeline)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [device])[0]

        img = data['img'][0]
        x = model.extract_feat(img)

        bbox_feat = model.roi_head.bbox_roi_extractor(
            x[:model.roi_head.bbox_roi_extractor.num_inputs], roi)
        
        return bbox_feat

    def __average_pooling(self, x):
        """
        Args:
        x: dtype: numpy.ndarray
        """
        x = self.avg_pool(x)
        x = x.flatten(1)
        return x

if __name__ == '__main__':

    '''
    This is the pre-processing process for video features with manual selected keyframes and speaker annotations.
    Note that Step 2 and Step 4 need manually select keyframes and speaker ids. The selected results can be seen in path_to_video_data/human_annotations.json.
    '''
    update_ids = ['S06_E08_555','S06_E01_134','S05_E16_433']

    args = parse_arguments()
    
    '''
    Step 1: Generate screenshots from raw videos.
    '''

    # generate_screenshots(args)
    
    '''
    Step 2: Generate bbox of frames with Faster R-CNN.
    '''
    # generate_frames_bbox(args)

    '''
    Step 3: Manually select the bbox id of the speaker and produce speaker_annotations.json.
    (The selected speaker ids can be seen in path_to_video_data/human_annotations.json).

    The format of 'speaker_annotations.json' is like:
    {
        "S05_E14_102_20_4.jpg": {
            'id': 0,
        },
        "xxx.jpg":{
            'id': xxx,
        }
        ...
    }
    '''

    '''
    Step 4: Generate the bbox information of the corresponding speaker_id.
    '''
    # generate_speaker_bbox(args)

    '''
    Step 5: Generate video features
    '''
    args.use_TalkNet = True
    args.video_data_path = '/home/sharing/Datasets/L-M-8081/L-MIntRec/video_data'
    args.video_feats_path = 'video_feats.pkl'
    args.TalkNet_speaker_path = '/home/sharing/disk1/zhanghanlei/L-M_detect_speaker_8081'
    video_data = VideoFeature(args)



