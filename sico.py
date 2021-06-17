#!/usr/bin/env python3
# © 2021: github.com/th-under
#


import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe as mp
import time
from scipy import spatial

import global_vars as gv

from helpers import orientation
from helpers import eye_overlay
from helpers import stack_images




cap = cv2.VideoCapture(gv.DEVICE_IN)
cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 1)


print('Hit enter when ready to capture tweak images.')
print('Combine upward, downward, left, and right head motions until webcam')
print('switches off after some seconds while looking straight into the cam.')
input()

# capture reference eyes
tweak_imgs = []
tweak_coords = []

current_t = int(10 * time.time())

for frame in range(80):
    
    # wait until next full 0.1 s have passed 
    current_t += 1
    while int(10 * time.time()) < current_t:
        time.sleep(0.01)
    
    tweak_coords.append({})
    
    success, img = cap.read()
    
    if success:
        tweak_imgs.append(img)
        
        results = faceMesh.process(img)
        
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                for lm in gv.EYE_CIRCLE_R + gv.EYE_CIRCLE_L + gv.FACE:
                    tweak_coords[frame][lm] = np.array([
                        round(faceLms.landmark[lm].x * cam_w), 
                        round(faceLms.landmark[lm].y * cam_h)])

    cv2.imshow("Reference eyes", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()




tweak_orient = np.full((len(tweak_coords),2),np.nan)

for i, frame in enumerate(tweak_coords):
    tweak_orient[i] = orientation(frame)



    
# improvements: axis labels, ...
plt.scatter(tweak_orient[:,0], tweak_orient[:,1])
        

# which tweak_image should be used for which head orientation?
# map to discrete values to provide look up table
# tweak_table[left-right orientation, down-up orientation,:]
# tweak_table[:,:,0] is the distance (error) of mapped vs. existing image
# tweak_table[:,:,1] is the tweak_image index closest to the given orientations 

tweak_table = np.full((int(1/gv.TWEAK_DENS + 1), int(1/gv.TWEAK_DENS + 1),2),np.nan)

for lr_i, lr in enumerate(np.arange(0., 1.001, gv.TWEAK_DENS)):
    for du_j, du in enumerate(np.arange(0., 1.001, gv.TWEAK_DENS)):
        
        tweak_table[lr_i,du_j] = spatial.KDTree(tweak_orient).query([lr, du])
        
        
        
        
# capture real face and replace eye region

cap = cv2.VideoCapture(gv.DEVICE_IN)
prev_live_coords = {}

while True:
    
    success, img = cap.read()
    
    if success:
        
        live_coords = {}
        
        results = faceMesh.process(img)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                for lm in gv.EYE_CIRCLE_R + gv.EYE_CIRCLE_L + gv.FACE:
                    live_coords[lm] = np.array([
                        round(faceLms.landmark[lm].x * cam_w), 
                        round(faceLms.landmark[lm].y * cam_h)])
        
        
        result_img, tweaked = eye_overlay(img, live_coords, prev_live_coords, tweak_imgs, tweak_coords, tweak_table)

        prev_live_coords = live_coords
        
        stacked_image = stack_images(img, result_img, 0.3333)
        
        cv2.putText(stacked_image,'original image', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(stacked_image,'tweaked eyes', (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if tweaked:
            cv2.putText(stacked_image,'tweaked', (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
            
        if gv.MODE == 'demo':
            cv2.imshow("Demo", stacked_image)
            if cv2.waitKey(1) == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

        if gv.MODE == 'live':
            cv2.imshow("live", result_img)
            if cv2.waitKey(1) == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break




