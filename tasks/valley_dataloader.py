import numpy as np
import os
import pandas as pd
import torch
import decord
import timeit
import pickle
import json
import random
import cv2
import ffmpeg

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.autograd.variable import Variable
from torchvision import transforms
from decord import VideoReader, cpu, gpu
from datetime import timedelta
from torch.utils.data import DistributedSampler
from torch.utils.data._utils.collate import default_collate


# print('imported!!!')
# os.environ['DECORD_EOF_RETRY_MAX'] = '40960'
decord.bridge.set_bridge('torch')
# test_remove = ['084k_RL3ApU_000109_000119.mp4', '2xWiEVNUvhE_000064_000074.mp4', '305P2f9_lko_004145_004155.mp4',
#                'B4bn9G6__sY_000086_000096.mp4', 'BvBVQmm2RcM_000082_000092.mp4', 'CxjipYE57Yo_000199_000209.mp4',
#                'IhanWvpHGu8_001243_001253.mp4', 'Lw14NH9kAqE_000759_000769.mp4',
#                ' XFkykETgkoo_002967_002977.mp4', 'jJFqy6yiXzQ_000024_000034.mp4',
#                'kinMMqkswUk_000120_000130.mp4', 'y7cYaYX4gdw_000047_000057.mp4']


def parse_json(data):
    with open(data, 'r') as file:
        data = json.load(file)

    videos = []
    text = []

    for x in data:
        try:
            video = x.get('video', None)
            conversation_list = x.get('conversations', [])
            
            if len(conversation_list) > 1:
                label = conversation_list[1].get('value', None)
            else:
                label = None
            
            if video and label:
                videos.append(video)
                text.append(label)
            else:
                print(f"Skipping due to missing video or label: {x}")
        except Exception as e:
            print(f"Error processing {x}: {e}")

    print(f"Number of videos: {len(videos)}")
    print(f"Number of text: {len(text)}")

    return videos, text


def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, text, index = zip(*batch)
    inputs = [x for x in inputs if x is not None]
    text = [x for x in text if x is not None]
    index = [x for x in index if x is not None]
#print(f'type of index start of collate: {type(index)}, value: {index}')

    inputs, text, index= (
        default_collate(inputs),
        default_collate(text),
        default_collate(index),
    )
#print(f'type of index end of collate: {type(index)}, value: {index}')
#print(f'type of index: {type(index)}')
    return inputs, text, index


class CustomBatchSampler(DistributedSampler):
    r"""Yield a mini-batch of indices. The sampler will drop the last batch of
            an image size bin if it is not equal to ``batch_size``

    Args:
        examples (dict): List from dataset class.
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, data, num_replicas, rank, shuffle):
        #print('init method')
        super().__init__(data, num_replicas, rank, shuffle)
        #self.batch_size = batch_size
        self.data = data
        if shuffle:
            random.shuffle(self.data)


    def __iter__(self):
        batch = []
        #num_frames = random.choice([4, 8, 16, 32, 64])
        num_frames = 8
        batch_switch = {
            4: 32,
            8: 16,
            16: 16,
            32: 8,
            64: 4
        }
        for index, sample in enumerate(self.data):
            batch_size = batch_switch[num_frames]
            batch.append([index, num_frames])

            if len(batch) == batch_size:
                #print(f'batch: {batch} batch type: {type(batch)}')
                print(f'batch size: {batch_size}, index: {index}')
                yield batch
                #num_frames = random.choice([4, 8, 16, 32, 64])
                num_frames = 8
                batch = []

    def __len__(self):
        return len(self.data)

    def set_epoch(self, epoch):
        if self.shuffle:
            random.shuffle(self.data)

class VLDL(Dataset):
    def __init__(self, data_split, num_frames, flexible):
        self.text = []
        self.num_frames = num_frames
        self.flexible = flexible
        self.root = '/sqsh/Video-LLaVA/valley/'
        self.data_split = data_split
        self.media_type = 'video'

        
        #self.data = open('/home/fvidal/VideoMamba/videos.txt', 'r').read().splitlines()
        self.data, self.text = parse_json('/home/fvidal/VideoMamba/videomamba/video_mm/tasks/chat.json')
        
        if data_split == 'train':
            self.data = self.data[:int(0.8 * len(self.data))]
        else:
            self.data = self.data[int(0.8 * len(self.data)):]

        # 100 samples for training
        self.data = self.data[:385000]

        #print(f'length of data: {len(self.data)}')
        #self.text = open('/home/fvidal/VideoMamba/text.txt', 'r').read().splitlines()
        #print(f'length of text: {len(self.text)}')

        
        #print(len(self.data), self.data[0], len(self.text), self.text[0])
        #self.data = self.data[3060:]
        #print(len(self.data), self.data[0], len(self.text), self.text[0])
        self.transforms = transforms.Compose([
            # need to find normalization values for valley
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize(256),
            transforms.CenterCrop(size=(224, 224))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #print('test 01')
        #print(f'flexible: {self.flexible}')
        #print(f'index; {index}')
        #if self.flexible:
        index, num_frames = index
        '''else:
            num_frames = self.num_frames
            '''
        if self.data_split == 'train':
            vid_path = self.data[index]
            label = vid_path.split('/')[0]

            vid_path = os.path.join(self.root, vid_path)
            #print(vid_path)
            try:
                vr = VideoReader(vid_path)
            except Exception:
                return None, None, None 
            frame_indexer = np.linspace(0, len(vr) - 1, num_frames)
            frames = vr.get_batch(frame_indexer)
            frames = frames.permute(0, 3, 1, 2) / 255.
            frames = self.transforms(frames)#.permute(1, 0, 2, 3)
            #torch.save(frames, 'frames.pt')
            #print(f'label at index {self.text[index]}')
            label = self.text[index]
            #print(f'frame shape: {frames.shape}')
            #label = self.text.index(label)


        elif self.data_split == 'test':
            vid_path = self.data[index]
            vid_path = os.path.join(self.root, vid_path)

            try:
                vr = VideoReader(vid_path)
            except Exception:
                return None, None
            frame_indexer = np.linspace(0, len(vr) - 1, num_frames)
            frames = vr.get_batch(frame_indexer)
            frames = frames.permute(0, 3, 1, 2) / 255.
            frames = self.transforms(frames).permute(1, 0, 2, 3)
            label = self.text[index]
            #print(f'type of index in getitem: {type(index)}')
        #torch.save(frames, f'/frames/frame{index}.pth')
        return frames, label, index


if __name__ == '__main__':
    shuffle = False
    dataloader_gen = VLDL('train', 8, True)
    cb_sampler = CustomBatchSampler(dataloader_gen.data, batch_size=8, num_replicas=1, rank=0, shuffle=True)
    dataloader = DataLoader(dataloader_gen, num_workers=0, batch_sampler=cb_sampler, collate_fn=multiple_samples_collate)
    '''
    for frames, label, index in tqdm(dataloader):
        #print(f'frame shape: {frames.shape}, label: {label}, index: {index}')
        save_dir = f'/frames/img{index}.pth'
        torch.save(frames, save_dir)
        print(f'saved frames at {save_dir} with label: {label}')
        '''
