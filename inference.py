import concurrent.futures
import glob

# from utils.deep_speech import DeepSpeech
import logging
import os
import random
import subprocess
import time
from collections import OrderedDict
from timeit import default_timer

import cv2
import numpy as np
import torch
import sys
from config.config import DINetInferenceOptions
from models.DINet import DINet
from utils.data_processing import compute_crop_radius, load_landmark_openface
from utils.wav2vec import Wav2VecFeatureExtractor
from utils.wav2vecDS import Wav2vecDS
from pathlib import Path

import pygame as pg
from pygame._sdl2 import (
    get_audio_device_names,
    AudioDevice,
    AUDIO_S16,
    AUDIO_ALLOW_FORMAT_CHANGE,
)
# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create an instances of the Wav2VecFeatureExtractor and Wav2vecDS classes
feature_extractor = Wav2VecFeatureExtractor()
audio_mapping = Wav2vecDS()

from pathlib import Path
from openai import OpenAI



class State:
    def __init__(self):
        self.recording = False
        self.sound_chunks = []
        self.audio = None


def setup_audio(state: State):
    pg.mixer.pre_init(44100, -16, 1, 2048)
    pg.init()

    names = get_audio_device_names(True)
    # Add a flag to control recording

    """This is called in the sound thread."""

    def callback(audiodevice, audiomemoryview):
        if state.recording:
            state.sound_chunks.append(bytes(audiomemoryview))

    # set_post_mix(callback)

    state.audio = AudioDevice(
        devicename=names[0],
        iscapture=True,
        frequency=44100 * 2,
        audioformat=AUDIO_S16,
        numchannels=1,
        chunksize=2048,
        allowed_changes=AUDIO_ALLOW_FORMAT_CHANGE,
        callback=callback,
    )
    state.audio.pause(0)


def is_speech_start(sound_chunks, threshold=1):
    return sound_chunks[-1] >= threshold


def setup_display():
    pg.init()
    WIDTH = 500
    HEIGHT = 500

    windowSurface = pg.display.set_mode((WIDTH, HEIGHT), 32)
    return windowSurface


# Frames extraction took 29.91 sec
def extract_frames_from_video(video_path, save_dir):
    videoCapture = cv2.VideoCapture(video_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        print(
            "warning: the input video is not 25 fps, it would be better to trans it to 25 fps!"
        )
    frames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))

    os.makedirs(save_dir, exist_ok=True)
    # Construct the ffmpeg command
    ffmpeg_command = ["ffmpeg", "-i", video_path, os.path.join(save_dir, "%06d.png")]

    # Run the ffmpeg command
    subprocess.run(
        ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )

    return frame_width, frame_height

def check_frame_path(video_dir):
    temp = cv2.imread(video_dir+"/000001.png")
    return temp.shape[1], temp.shape[0]    

def convert_opencv_img_to_pygame(opencv_image):
    """
    OpenCVの画像をPygame用に変換.

    see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
    """
    opencv_image = opencv_image[:,:,::-1]  # OpenCVはBGR、pygameはRGBなので変換してやる必要がある。
    shape = opencv_image.shape[1::-1]  # OpenCVは(高さ, 幅, 色数)、pygameは(幅, 高さ)なのでこれも変換。
    pygame_image = pg.image.frombuffer(opencv_image.tostring(), shape, 'RGB')

    return pygame_image
def a2m_preprocess(img_dir, video_frame_path_list, audio_path):
    # load config
    opt = DINetInferenceOptions().parse_args()
    # if not os.path.exists(opt.source_video_path):
    #     raise ValueError("wrong video path : {}".format(opt.source_video_path))
    if not os.path.exists(opt.source_openface_landmark_path):
        raise ValueError(
            "wrong openface stats path : {}".format(opt.source_openface_landmark_path)
        )

    # extract frames from source video
    # logging.info("extracting frames from video: %s", opt.source_video_path)
    # start_time = time.time()
    # video_frame_dir = opt.source_video_path.replace(".mp4", "")
    # if not os.path.exists(video_frame_dir):
    #     os.mkdir(video_frame_dir)
    video_size = check_frame_path(img_dir)
    #end_time = time.time()
    #logging.info(f"Frames extraction took {end_time - start_time:.2f} sec.")

    # extract audio features using Hubert Model from Pytorch
    logging.info("extracting audio speech features from : %s", audio_path)
    start_time = time.time()
    ds_feature = feature_extractor.compute_audio_feature(
        audio_path
    )  

    logging.info("Mapping Audio features")
    start_time_mapping = time.time()
    ds_feature = audio_mapping.mapping(ds_feature)
    end_time_mapping = time.time()
    logging.info(
        f"Mapping audio features took {end_time_mapping - start_time_mapping:.2f} sec."
    )
    res_frame_length = ds_feature.shape[0]
    ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode="edge")

    end_time = time.time()
    logging.info(f"Audio features extraction took {end_time - start_time:.2f} sec.")

    # load facial landmarks
    logging.info(
        "loading facial landmarks from : %s", opt.source_openface_landmark_path
    )
    if not os.path.exists(opt.source_openface_landmark_path):
        raise ValueError(
            "wrong facial landmark path :%s", opt.source_openface_landmark_path
        )
    video_landmark_data = load_landmark_openface(
        opt.source_openface_landmark_path
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
    res_video_frame_path_list_pad = (
        [video_frame_path_list_cycle[0]] * 2
        + res_video_frame_path_list
        + [video_frame_path_list_cycle[-1]] * 2
    )
    res_video_landmark_data_pad = np.pad(
        res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode="edge"
    )
    assert (
        ds_feature_padding.shape[0]
        == len(res_video_frame_path_list_pad)
        == res_video_landmark_data_pad.shape[0]
    )
    pad_length = ds_feature_padding.shape[0]

    # randomly select 5 reference images
    logging.info("selecting five reference images")
    ref_img_list = []
    resize_w = int(opt.mouth_region_size + opt.mouth_region_size // 4)
    resize_h = int((opt.mouth_region_size // 2) * 3 + opt.mouth_region_size // 8)
    ref_index_list = random.sample(range(5, len(res_video_frame_path_list_pad) - 2), 5)
    for ref_index in ref_index_list:
        crop_flag, crop_radius = compute_crop_radius(
            video_size, res_video_landmark_data_pad[ref_index - 5 : ref_index, :, :]
        )
        if not crop_flag:
            raise ValueError(
                "our method cannot handle videos with large changes in facial size!!"
            )
        crop_radius_1_4 = crop_radius // 4
        ref_img = cv2.imread(res_video_frame_path_list_pad[ref_index - 3])[:, :, ::-1]
        ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
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
        ref_img_crop = cv2.resize(ref_img_crop, (resize_w, resize_h))
        ref_img_crop = ref_img_crop / 255.0
        ref_img_list.append(ref_img_crop)
    ref_video_frame = np.concatenate(ref_img_list, 2)
    ref_img_tensor = (
        torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()
    )

    # load pretrained model weight
    logging.info("loading pretrained model from: %s", opt.pretrained_clip_DINet_path)
    model = DINet(opt.source_channel, opt.ref_channel, opt.audio_channel).cuda()
    if not os.path.exists(opt.pretrained_clip_DINet_path):
        raise ValueError(
            "wrong path of pretrained model weight: %s", opt.pretrained_clip_DINet_path
        )
    state_dict = torch.load(opt.pretrained_clip_DINet_path)["state_dict"]["net_g"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model, pad_length, video_size, opt, ds_feature_padding,ref_img_tensor, resize_w,resize_h, res_video_landmark_data_pad, res_video_frame_path_list_pad


def a2m(model, idx, video_size, opt, ds_feature_padding,ref_img_tensor, resize_w,resize_h, res_video_landmark_data_pad, res_video_frame_path_list_pad):
    ############################################## inference frame by frame ##############################################
    logging.info("rendering result video")
    # if not os.path.exists(opt.res_video_dir):
    #     os.mkdir(opt.res_video_dir)
    # res_video_path = os.path.join(
    #     opt.res_video_dir,
    #     os.path.basename(opt.source_video_path)[:-4] + "_facial_dubbing.mp4",
    # )
    # if os.path.exists(res_video_path):
    #     os.remove(res_video_path)
    # res_face_path = res_video_path.replace("_facial_dubbing.mp4", "_synthetic_face.mp4")
    # if os.path.exists(res_face_path):
    #     os.remove(res_face_path)
    clip_end_index = idx +5
    print(video_size)
    logging.info("synthesizing frame %d", clip_end_index - 5)
    crop_flag, crop_radius = compute_crop_radius(
        video_size,
        res_video_landmark_data_pad[clip_end_index - 5 : clip_end_index, :, :],
        random_scale=1.05,
    )
    if not crop_flag:
        raise (
            "our method can not handle videos with large change of facial size!!"
        )
    crop_radius_1_4 = crop_radius // 4
    frame_data = cv2.imread(res_video_frame_path_list_pad[clip_end_index - 3])[
        :, :, ::-1
    ]
    frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]
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
        crop_frame_data, (resize_w, resize_h)
    )  # [32:224, 32:224, :]
    crop_frame_data = crop_frame_data / 255.0
    crop_frame_data[
        opt.mouth_region_size // 2 : opt.mouth_region_size // 2
        + opt.mouth_region_size,
        opt.mouth_region_size // 8 : opt.mouth_region_size // 8
        + opt.mouth_region_size,
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
        torch.from_numpy(ds_feature_padding[clip_end_index - 5 : clip_end_index, :])
        .permute(1, 0)
        .unsqueeze(0)
        .float()
        .cuda()
    )
    with torch.no_grad():
        pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
        pre_frame = (
            pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
        )
    
        # videowriter_face.write(pre_frame[:, :, ::-1].copy().astype(np.uint8))
        # cv2.imshow("a",pre_frame[:, :, ::-1].copy().astype(np.uint8))
        # cv2.waitKey(1)
    pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w, crop_frame_h))
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
    return frame_data[:, :, ::-1]#pre_frame[:, :, ::-1].copy().astype(np.uint8)
    #     videowriter.write(frame_data[:, :, ::-1])
    # videowriter.release()
    # videowriter_face.release()
    # video_add_audio_path = res_video_path.replace(".mp4", "_add_audio.mp4")
    # if os.path.exists(video_add_audio_path):
    #     os.remove(video_add_audio_path)
    # cmd = "ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {}".format(
    #     res_video_path, opt.driving_audio_path, video_add_audio_path
    # )
    # subprocess.call(cmd, shell=True)
    #end_process = default_timer()
    #logging.info(f"Video generation took {end_process - start_process:.2f} sec.")

def device(data_root):

    state = State()
    windowSurface = setup_display()
    #setup_audio(state)
    clock = pg.time.Clock()
    vid_name = "/out2/"
    ori_vid_name = "/out/"
    video_path = data_root+ori_vid_name
    img_path = data_root+vid_name


    client = OpenAI()
    inp = input("Input the Questions: ")
    inp = inp + " Please answer under 20 words, and start with 没问题"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": inp}]
    )
    assistant_res = response.choices[0].message.content

    print(assistant_res)
    speech_file_path = data_root+"/speech.mp3"
    
    response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input=assistant_res
    )
    response.stream_to_file(speech_file_path)

    audio_path = data_root+"/speech.mp3"

    image_list = sorted(glob.glob(img_path+"*.png"))
    ori_vid_list = sorted(glob.glob(video_path+"*.png"))
    model, pad_length, video_size, opt, ds_feature_padding,ref_img_tensor, resize_w,resize_h, res_video_landmark_data_pad, res_video_frame_path_list_pad = a2m_preprocess(img_path, image_list, audio_path)
    
    f = 0
    running = True
    print("generated " + str(pad_length) + " frames")
    pg.mixer.music.load(audio_path)
    pg.mixer.music.play()
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

            elif event.type == pg.KEYDOWN:
                # Quit when the user presses the ESC key
                if event.key == pg.K_ESCAPE:
                    running = False
                # Start recording when SPACE key is pressed
                elif event.key == pg.K_SPACE:
                    state.sound_chunks = []
                    state.recording = True

            elif event.type == pg.KEYUP:
                # Stop recording when SPACE key is released
                if event.key == pg.K_SPACE:
                    state.recording = False
                    sound_data = pg.mixer.Sound(buffer=b"".join(state.sound_chunks))
                    sound = pg.mixer.Sound(buffer=sound_data)
                    sound.play()
        if f < pad_length - 2:
            img = a2m(model, f, video_size, opt, ds_feature_padding,ref_img_tensor, resize_w,resize_h, res_video_landmark_data_pad, res_video_frame_path_list_pad)
        else:
            img = cv2.imread(ori_vid_list[f-3])[200:700,300:800]
        #ori_img = cv2.imread(ori_vid_list[f])
        #ori_img[200:700,300:800] = img
        #ori_img = cv2.resize(ori_img, dsize=None, fx=0.5, fy=0.5)
        #ori_vid_list = cv2.resize()
        img = convert_opencv_img_to_pygame(img)

        windowSurface.blit(img, (0, 0))
        pg.display.flip()
        f += 1
        f = f % len(image_list)
        clock.tick(25)
    pg.display.quit()
    pg.quit()
    exit()

if __name__ == "__main__":
    device("asserts/examples/")
    #a2m()
