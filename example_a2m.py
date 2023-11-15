import glob
import cv2 
import numpy as np
from a2m import A2M
from openai import OpenAI
from a2m import convert_opencv_img_to_pygame
import time
import pygame as pg

#generate audio file using TTS
audio_file_path = "./output.mp3"
# client = OpenAI()
# response = client.audio.speech.create(model="tts-1",
#                                       voice="nova",
#                                       input="This is for testing A2M module.")
# response.stream_to_file(audio_file_path)

#get frame folder and img list
img_path = "asserts/examples/out_kawaii/"
image_list = sorted(glob.glob(img_path+"*.png"))

#load pygame
pg.init()
windowSurface = pg.display.set_mode((500, 500), 32)
clock = pg.time.Clock()
pg.mixer.music.load(audio_file_path)

#initia A2M
a2m = A2M()
a2m.initiate()


for i in range(5):
    starttime = time.time()
    a2m.load(img_path, image_list, audio_file_path)
    f = 0
    print( time.time() -starttime)
    pg.mixer.music.play()
    while f< a2m.pad_length-2:
        img = a2m.gen_frame(f)
        img = convert_opencv_img_to_pygame(img)
        windowSurface.blit(img, (0, 0))
        pg.display.flip()
        f += 1
        clock.tick(100)
    


pg.display.quit()
pg.quit()
exit()