# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:47:24 2020

@author: ABC
"""

# import required packages
import requests
import cv2
import os
 
from imutils import paths
 
url_path = open('hawamahal_d').read().strip().split('\n')
total = 0
 
if not os.path.exists('images'):
    os.mkdir('images')
image_path = 'images'
name = 'lotustemple.'
 
for url in url_path:
    try:
        req = requests.get(url, timeout=60)
 
        file_path = os.path.sep.join([image_path, '{}.jpg'.format(
            name+str(total))]
        )
        file = open(file_path, 'wb')
        file.write(req.content)
        file.close()
 
        print('Downloaded {}'.format(file_path))
        total += 1
 
    except:
        print('Could not download {}. Downloading next file'.format(file_path))