import logging
from config.config import DINetInferenceOptions
import os
import torch
import random
import subprocess
import time
from collections import OrderedDict
import cv2
import numpy as np
import torch
import sys
from config.config import DINetInferenceOptions
from models.DINet import DINet
from utils.data_processing import compute_crop_radius, load_landmark_openface
from utils.wav2vec import Wav2VecFeatureExtractor
from utils.wav2vecDS import Wav2vecDS
import pygame as pg 
def convert_opencv_img_to_pygame(opencv_image):
    """
    OpenCVの画像をPygame用に変換.

    see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
    """
    opencv_image = opencv_image[:,:,::-1]  # OpenCVはBGR、pygameはRGBなので変換してやる必要がある。
    shape = opencv_image.shape[1::-1]  # OpenCVは(高さ, 幅, 色数)、pygameは(幅, 高さ)なのでこれも変換。
    pygame_image = pg.image.frombuffer(opencv_image.tostring(), shape, 'RGB')

    return pygame_image


class A2M:
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    def __init__(self, name="A2M"):
        self.model = None
        self.pad_length = 0
        self.video_size = (1920,1080)
        # load config
        self.opt = DINetInferenceOptions().parse_args()
        self.ds_feature_padding = None
        self.ref_img_tensor = None
        self.resize_w = 500
        self.resize_h = 500
        self.res_video_landmark_data_pad = None
        self.res_video_frame_path_list_pad = None
        # Create an instances of the Wav2VecFeatureExtractor and Wav2vecDS classes
        self.feature_extractor = Wav2VecFeatureExtractor()
        self.audio_mapping = Wav2vecDS()

    #check first frame and return frame shape
    def check_frame_path(self, video_dir):
        temp = cv2.imread(video_dir+"/000001.png")
        return temp.shape[1], temp.shape[0]   
    
    def initiate(self):


        # load pretrained model weight
        logging.info("loading pretrained model from: %s", self.opt.pretrained_clip_DINet_path)
        self.model = DINet(self.opt.source_channel, self.opt.ref_channel, self.opt.audio_channel).cuda()
        if not os.path.exists(self.opt.pretrained_clip_DINet_path):
            raise ValueError(
                "wrong path of pretrained model weight: %s", self.opt.pretrained_clip_DINet_path
            )
        state_dict = torch.load(self.opt.pretrained_clip_DINet_path)["state_dict"]["net_g"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def load(self, img_dir, video_frame_path_list, audio_path):

        #extract video size from 1 frame
        self.video_size = self.check_frame_path(img_dir)

        # extract audio features using Hubert Model from Pytorch
        logging.info("extracting audio speech features from : %s", audio_path)
        start_time = time.time()
        ds_feature = self.feature_extractor.compute_audio_feature(
            audio_path
        )  
        # map audio feature 
        logging.info("Mapping Audio features")
        start_time_mapping = time.time()
        ds_feature = self.audio_mapping.mapping(ds_feature)
        end_time_mapping = time.time()
        logging.info(
            f"Mapping audio features took {end_time_mapping - start_time_mapping:.2f} sec."
        )
        res_frame_length = ds_feature.shape[0]
        self.ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode="edge")

        end_time = time.time()
        logging.info(f"Audio features extraction took {end_time - start_time:.2f} sec.")

        # load facial landmarks
        source_openface_landmark_path = img_dir[:-1] + '.csv'
        logging.info(
            "loading facial landmarks from : %s", source_openface_landmark_path
        )

        if not os.path.exists(source_openface_landmark_path):
            raise ValueError(
                "wrong facial landmark path :%s", source_openface_landmark_path
            )
        video_landmark_data = load_landmark_openface(
            source_openface_landmark_path
        ).astype(np.int)

        # align frame with driving audio
        logging.info("aligning frames with driving audio")
        print(len(video_frame_path_list))
        if len(video_frame_path_list) != video_landmark_data.shape[0]:
            raise ValueError("video frames are misaligned with detected landmarks")
        video_frame_path_list_cycle = video_frame_path_list + video_frame_path_list[::-1]
        video_landmark_data_cycle = np.concatenate(
            [video_landmark_data, np.flip(video_landmark_data, 0)], 0
        )
        video_frame_path_list_cycle_length = len(video_frame_path_list_cycle)
        if video_frame_path_list_cycle_length >= res_frame_length:
            res_video_frame_path_list = video_frame_path_list_cycle[:res_frame_length]
            res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
        else:
            divisor = res_frame_length // video_frame_path_list_cycle_length
            remainder = res_frame_length % video_frame_path_list_cycle_length
            res_video_frame_path_list = (
                video_frame_path_list_cycle * divisor
                + video_frame_path_list_cycle[:remainder]
            )
            res_video_landmark_data = np.concatenate(
                [video_landmark_data_cycle] * divisor
                + [video_landmark_data_cycle[:remainder, :, :]],
                0,
            )
        self.res_video_frame_path_list_pad = (
            [video_frame_path_list_cycle[0]] * 2
            + res_video_frame_path_list
            + [video_frame_path_list_cycle[-1]] * 2
        )
        self.res_video_landmark_data_pad = np.pad(
            res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode="edge"
        )
        assert (
            self.ds_feature_padding.shape[0]
            == len(self.res_video_frame_path_list_pad)
            == self.res_video_landmark_data_pad.shape[0]
        )
        self.pad_length = self.ds_feature_padding.shape[0]

        # randomly select 1 reference images
        logging.info("selecting five reference images")
        ref_img_list = []
        self.resize_w = int(self.opt.mouth_region_size + self.opt.mouth_region_size // 4)
        self.resize_h = int((self.opt.mouth_region_size // 2) * 3 + self.opt.mouth_region_size // 8)
        ref_index_list = random.sample(range(5, len(self.res_video_frame_path_list_pad) - 2), 5)
        for ref_index in ref_index_list:
            crop_flag, crop_radius = compute_crop_radius(
                self.video_size, self.res_video_landmark_data_pad[ref_index - 5 : ref_index, :, :]
            )
            if not crop_flag:
                raise ValueError(
                    "our method cannot handle videos with large changes in facial size!!"
                )
                
            crop_radius_1_4 = crop_radius // 4
            ref_img = cv2.imread(self.res_video_frame_path_list_pad[ref_index - 3])[:, :, ::-1]
            ref_landmark = self.res_video_landmark_data_pad[ref_index - 3, :, :]
            ref_img_crop = ref_img[
                ref_landmark[29, 1]
                - crop_radius : ref_landmark[29, 1]
                + crop_radius * 2
                + crop_radius_1_4,
                ref_landmark[33, 0]
                - crop_radius
                - crop_radius_1_4 : ref_landmark[33, 0]
                + crop_radius
                + crop_radius_1_4,
                :,
            ]
            ref_img_crop = cv2.resize(ref_img_crop, (self.resize_w, self.resize_h))
            ref_img_crop = ref_img_crop / 255.0
            ref_img_list.append(ref_img_crop)
        ref_video_frame = np.concatenate(ref_img_list, 2)
        self.ref_img_tensor = (
            torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()
        )
       
    # inference the frames
    def gen_frame(self, idx):
        clip_end_index = idx +5
        logging.info("synthesizing frame %d", clip_end_index - 5)
        crop_flag, crop_radius = compute_crop_radius(
            self.video_size,
            self.res_video_landmark_data_pad[clip_end_index - 5 : clip_end_index, :, :],
            random_scale=1.05,
        )
        if not crop_flag:
            raise (
                "our method can not handle videos with large change of facial size!!"
            )
        crop_radius_1_4 = crop_radius // 4
        frame_data = cv2.imread(self.res_video_frame_path_list_pad[clip_end_index - 3])[
            :, :, ::-1
        ]
        frame_landmark = self.res_video_landmark_data_pad[clip_end_index - 3, :, :]

        crop_frame_data = frame_data[
            frame_landmark[29, 1]
            - crop_radius : frame_landmark[29, 1]
            + crop_radius * 2
            + crop_radius_1_4,
            frame_landmark[33, 0]
            - crop_radius
            - crop_radius_1_4 : frame_landmark[33, 0]
            + crop_radius
            + crop_radius_1_4,
            :,
        ]
        crop_frame_h, crop_frame_w = crop_frame_data.shape[0], crop_frame_data.shape[1]
        crop_frame_data = cv2.resize(
            crop_frame_data, (self.resize_w, self.resize_h)
        )  # [32:224, 32:224, :]
        crop_frame_data = crop_frame_data / 255.0
        
        #crop the frame by mouth region
        crop_frame_data[
            self.opt.mouth_region_size // 2 : self.opt.mouth_region_size // 2
            + self.opt.mouth_region_size,
            self.opt.mouth_region_size // 8 : self.opt.mouth_region_size // 8
            + self.opt.mouth_region_size,
            :,
        ] = 0
        crop_frame_tensor = (
            torch.from_numpy(crop_frame_data)
            .float()
            .cuda()
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        deepspeech_tensor = (
            torch.from_numpy(self.ds_feature_padding[clip_end_index - 5 : clip_end_index, :])
            .permute(1, 0)
            .unsqueeze(0)
            .float()
            .cuda()
        )
        with torch.no_grad():
            pre_frame = self.model(crop_frame_tensor, self.ref_img_tensor, deepspeech_tensor)
            pre_frame = (
                pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
            )
        # resize and blend the frame back to the original image.
        pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w, crop_frame_h))

        #return pre_frame_resize
        frame_data[
            frame_landmark[29, 1]
            - crop_radius : frame_landmark[29, 1]
            + crop_radius * 2,
            frame_landmark[33, 0]
            - crop_radius
            - crop_radius_1_4 : frame_landmark[33, 0]
            + crop_radius
            + crop_radius_1_4,
            :,
        ] = pre_frame_resize[: crop_radius * 3, :, :]
        return frame_data[:, :, ::-1]



