# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:07:30 2020

@author: ABC
"""

for imagePath in paths.list_images('images'):
    delete_image = False
 
    try:
        image = cv2.imread(imagePath)
 
        if image is None:
            delete_image = True
 
    # if OpenCV cannot load the image
    except:
        delete_image = True
 
    if delete_image:
        print('Deleting {}'.format(imagePath))
        os.remove(imagePath)