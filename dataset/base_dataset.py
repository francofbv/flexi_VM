import logging
import os
import random
from torch.utils.data import Dataset
from dataset.utils import load_image_from_path
import json

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False

logger = logging.getLogger(__name__)


f = open('/home/fvidal/webvid_10m_segments.json')
segments = json.load(f)

def fix_data_path(data_path):
    path = '/datasets/WebVid/data/videos/'
    name = data_path[17:data_path.index('.')]
    fixed = os.path.join(path, segments[name]['segment'], f'{name}.mp4') if name in segments else 'Empty_path'
    return fixed


class ImageVideoBaseDataset(Dataset):
    """Base class that implements the image and video loading methods"""

    media_type = "video"





    def __init__(self):
        assert self.media_type in ["image", "video"]
        self.data_root = None
        self.anno_list = (
            None  # list(dict), each dict contains {"image": str, # image or video path}
        )
        self.transform = None
        self.video_reader = None
        self.num_tries = None
        self.trimmed30 = False

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')

    def __getitem__(self, index):
        raise NotImplementedError

    def __len_(self):
        raise NotImplementedError

    def get_anno(self, index):
        """obtain the annotation for one media (video or image)

        Args:
            index (int): The media index.

        Returns: dict.
            - "image": the filename, video also use "image".
            - "caption": The caption for this file.

        """
        anno = self.anno_list[index]
        if self.data_root is not None:
            anno["image"] = os.path.join(self.data_root, anno["image"])
        return anno

    def load_and_transform_media_data(self, index, data_path):
        #data_path = fix_data_path(data_path)
        if self.media_type == "image":
            return self.load_and_transform_media_data_image(index, data_path)
        else:
            return self.load_and_transform_media_data_video(index, data_path)

    def load_and_transform_media_data_image(self, index, data_path):
        image = load_image_from_path(data_path, client=self.client)
        image = self.transform(image)
        return image, index

    def load_and_transform_media_data_video(self, index, data_path):
        #data_path = fix_data_path(data_path)
        for _ in range(self.num_tries):
            try:
                max_num_frames = self.max_num_frames if hasattr(self, "max_num_frames") else -1
                frames, frame_indices, video_duration = self.video_reader(
                    data_path, self.num_frames, self.sample_type, 
                    max_num_frames=max_num_frames, client=self.client,
                    trimmed30=self.trimmed30
                )
            except Exception as e:
                '''
                logger.warning(
                    f"Caught exception {e} when loading video {data_path}, "
                    f"randomly sample a new video as replacement"
                )
                '''
                index = random.randint(0, len(self) - 1)
                ann = self.get_anno(index)
                data_path = ann["image"]
                data_path = fix_data_path(data_path)
                continue
            # shared aug for video frames
            frames = self.transform(frames)
            #print('video successfully loaded')
            return frames, index
            '''
        else:

            raise RuntimeError(
                f"Failed to fetch video after {self.num_tries} tries. "
                f"This might indicate that you have many corrupted videos."
            )
            '''
