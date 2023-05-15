#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 18:38:52 2023

@author: pippala
"""

import cv2
import os
from moviepy.editor import ImageSequenceClip
from current_plan_with_heuristics import plan


os.chdir('/Users/pippala/Desktop/robotic/location-based-generative-master')

# Define the path to the folder containing the images
# # CLEVR_new_001689.png to CLEVR_new_000555.png
# image_files = ['CLEVR_new_001689.png', 'CLEVR_new_001683.png', 'CLEVR_new_000339.png', 'CLEVR_new_000345.png', 'CLEVR_new_000555.png']

# # CLEVR_new_000915.png to CLEVR_new_001341.png
# image_files  = ['CLEVR_new_000915.png', 'CLEVR_new_000916.png', 'CLEVR_new_001252.png', 'CLEVR_new_001258.png', 'CLEVR_new_001341.png']


# # CLEVR_new_001517.png to CLEVR_new_000867.png
# image_files  = ['CLEVR_new_001517.png', 'CLEVR_new_001515.png', 'CLEVR_new_001599.png', 'CLEVR_new_000927.png', 'CLEVR_new_000843.png', 'CLEVR_new_000867.png']


image_files = plan(1) #get w 

# we get 
# CLEVR_new_000632.png to CLEVR_new_001062.png
# CLEVR_new_000632.png
# CLEVR_new_001640.png
# CLEVR_new_001406.png
# CLEVR_new_001404.png
# CLEVR_new_001398.png
# CLEVR_new_001062.png


os.chdir('/Users/pippala/Desktop/robotic/location-based-generative-master/photorealistic-blocksworld-master/blocks-5-3/image')
# Load the images and create an ImageSequenceClip
clip = ImageSequenceClip(image_files, fps=0.8)

# Write the clip to a video file
clip.write_videofile('output_video.mp4', fps=0.8)
