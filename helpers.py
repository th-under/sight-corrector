
import cv2
import numpy as np

import global_vars as gv

def stack_images(img1, img2, ratio):
    '''
    places two 3 channel images (RGB or BGR) in one image while keeping the
    result image dimensions like that of the two (identical) originals.
    img1 is placed at the bottom left corner, img2 at the top right corner.
    Remaining image areas will be black.
    

    Parameters
    ----------
    img1 : array of int
        first image
    img2 : array of int
        second image
    ratio: float [0 ... 1]
        scales img1 size down proportionally in width and height

    Returns
    -------
    new image

    '''
    assert img1.shape == img2.shape
    
    img_w = img1.shape[1]
    img_h = img1.shape[0]
    
    # region names
    
    # lu | ru
    # --------
    # ll | rl
    
    ll_h = int(img_h * ratio)
    ll_w = int(img_w * ratio)
    lu_h = img_h - ll_h
    lu_w = ll_w
    ru_h = ll_h
    ru_w = img_w - lu_w
    rl_h = img_h - ru_h
    rl_w = ru_w
     
    
    img1_small = cv2.resize(img1, (ll_w, ll_h))
    black = np.zeros((lu_h, lu_w, 3), np.uint8)
    left_img = np.vstack((black, img1_small))
    
    img2_small = cv2.resize(img2, (rl_w, rl_h))
    black = np.zeros((ru_h, ru_w, 3), np.uint8)
    right_img = np.vstack((img2_small, black))
    
    result_img = np.hstack((left_img, right_img))
    
    return result_img


def brightness(img):
    '''
    Calculates the brightness of an image (rgb, bgr, or gray) but omitts
    pixels with black color assuming they are a mask and do not belong to the
    image.
    
    Parameters
    ----------
    img : array
 
    Returns
    -------
    float
        brightness of the image.

    '''
    img = img.astype('float')
    img[img == 0] = np.nan
    if len(img.shape) == 3:
        # Color
        return np.nanmean(np.linalg.norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)

def orientation(coords):
    '''
    Estimates the current orientation in vertical and horizontal direction of a
    convex body. The estimation is based on five 2D pixel coordinates (x, y) of
    the body. The landmarks span the horizontal (2 landmarks) and the vertical
    (2 landmarks) expansion of the body while the central landmark enables to
    determine the division of the expansions.
    These relative proportions serve as orientation estimation of the body.

    Parameters
    ----------
    coords : dict 
        landmark numbers are the keys (defined globally)
        values [x, y] are the float coordinates
 
    Returns
    -------
    orient: [float, float]
        orient[0] == 1 means 100% to the right, orient[1] == 1 means 100% up

    '''
   
    try: 
         
        orient = [np.linalg.norm(coords[gv.FACE[4]] - coords[gv.FACE[0]]) / \
                  np.linalg.norm(coords[gv.FACE[4]] - coords[gv.FACE[3]]),
            
                  np.linalg.norm(coords[gv.FACE[2]] - coords[gv.FACE[0]]) / \
                  np.linalg.norm(coords[gv.FACE[2]] - coords[gv.FACE[1]])]
            
    except:
        
        return [np.nan, np.nan]
    
    return orient


def eye_overlay(live_img, live_coords, prev_live_coords,
                tweak_imgs, tweak_coords, tweak_table):
    '''
    Replaces the eyes of an image (live_image) with those from another image
    selected from a bunch of images (tweak_imgs) with different head
    orientations.
    Use case are webmeetings. Eyes should seem as if they are looking directly
    into a webcam to avoid looking desinterested while the speaker's looking
    direction does not need not face to the webcam but to the monitor, e.g.
    
    Parameters
    ----------
    live_img : array
        live image in opencv bgr format.
    live_coords : dict
        landmark keys and pixel coordinates as values.
    prev_live_coords : dict
        live coordinates of previous video frame if available. Used to reduce
        flickering.
    tweak_imgs : list of array
        bunch of images with different head positions and eyes looking into the
        webcam. Those eyes of the bunch of images are selected of which the
        head shows the most similar orientation as the live images.
    tweak_coords : list of dict
        face landmark pixel coordinates of the tweak images.
    tweak_table : array
        DESCRIPTION.

    Returns
    -------
    live_img : array
        tweaked image.
    bool
        whether the return image is tweaked.

    '''
    
    assert live_img.shape == tweak_imgs[0].shape, "Images must be the same size."
    
    cam_h, cam_w, _ = live_img.shape
    
    # select best suited tweak_image
    live_orient = np.array(orientation(live_coords))
    
    if ~np.isnan(live_orient[0]) and \
        np.min(live_orient) >= 0 and \
        np.max(live_orient) <= 1:

        indices = (round(live_orient[0] / gv.TWEAK_DENS), 
                   round(live_orient[1] / gv.TWEAK_DENS))
    else:
        return live_img, False
    
    if tweak_table[indices][0] > gv.MAP_TOL:
        return live_img, False
    else:
        tweak_img    = tweak_imgs  [int(tweak_table[indices][1])]
        tweak_coords = tweak_coords[int(tweak_table[indices][1])]
    
    # affine transform tweak image to live image_coords, separately for both eyes
    
    tw_eye_lm_r = np.float32([tweak_coords[gv.EYE_CIRCLE_R[8]],
                              tweak_coords[gv.EYE_CIRCLE_R[4]],
                              tweak_coords[gv.EYE_CIRCLE_R[12]],
                              tweak_coords[gv.EYE_CIRCLE_R[0]]])
    
    lv_eye_lm_r = np.float32([live_coords[gv.EYE_CIRCLE_R[8]],
                              live_coords[gv.EYE_CIRCLE_R[4]],
                              live_coords[gv.EYE_CIRCLE_R[12]],
                              live_coords[gv.EYE_CIRCLE_R[0]]])

    tw_eye_lm_l = np.float32([tweak_coords[gv.EYE_CIRCLE_L[0]],
                              tweak_coords[gv.EYE_CIRCLE_L[4]],
                              tweak_coords[gv.EYE_CIRCLE_L[12]],
                              tweak_coords[gv.EYE_CIRCLE_L[8]]])
    
    lv_eye_lm_l = np.float32([live_coords[gv.EYE_CIRCLE_L[0]],
                              live_coords[gv.EYE_CIRCLE_L[4]],
                              live_coords[gv.EYE_CIRCLE_L[12]],
                              live_coords[gv.EYE_CIRCLE_L[8]]])
    
# some damping to avoid flickering 
    if len(prev_live_coords) != 0:
        prev_lv_eye_lm_r = np.float32([prev_live_coords[gv.EYE_CIRCLE_R[8]],
                                       prev_live_coords[gv.EYE_CIRCLE_R[4]],
                                       prev_live_coords[gv.EYE_CIRCLE_R[12]],
                                       prev_live_coords[gv.EYE_CIRCLE_R[0]]])
        prev_lv_eye_lm_l = np.float32([prev_live_coords[gv.EYE_CIRCLE_L[0]],
                                       prev_live_coords[gv.EYE_CIRCLE_L[4]],
                                       prev_live_coords[gv.EYE_CIRCLE_L[12]],
                                       prev_live_coords[gv.EYE_CIRCLE_L[8]]])
        
    
        if np.linalg.norm (lv_eye_lm_r - prev_lv_eye_lm_r) < gv.DAMPING and \
           np.linalg.norm (lv_eye_lm_l - prev_lv_eye_lm_l) < gv.DAMPING:
            
            lv_eye_lm_r = (lv_eye_lm_r + prev_lv_eye_lm_r) / 2
            lv_eye_lm_l = (lv_eye_lm_l + prev_lv_eye_lm_l) / 2 
        
     
    M_r = cv2.getPerspectiveTransform(tw_eye_lm_r, lv_eye_lm_r)
    M_l = cv2.getPerspectiveTransform(tw_eye_lm_l, lv_eye_lm_l)
    
    tw_img_r_transf = cv2.warpPerspective(tweak_img, M_r, (cam_w, cam_h))
    tw_img_l_transf = cv2.warpPerspective(tweak_img, M_l, (cam_w, cam_h))
    

    # define mask (from live_image coordinates)
    
    right_polygon = np.array([live_coords[i] for i in gv.EYE_CIRCLE_R])
    right_polygon = right_polygon.reshape((-1,1,2))
    
    left_polygon = np.array([live_coords[i] for i in gv.EYE_CIRCLE_L])
    left_polygon = left_polygon.reshape((-1,1,2))
    
    mask_r = np.ones((cam_h, cam_w, 3), np.uint8) * 255
    mask_r = cv2.fillPoly(mask_r, [right_polygon], (0,0,0))
    img2gray = cv2.cvtColor(mask_r,cv2.COLOR_BGR2GRAY)
    ret, mask_r = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_r = cv2.bitwise_not(mask_r)

    mask_l = np.ones((cam_h, cam_w, 3), np.uint8) * 255
    mask_l = cv2.fillPoly(mask_l, [left_polygon], (0,0,0))
    img2gray = cv2.cvtColor(mask_l,cv2.COLOR_BGR2GRAY)
    ret, mask_l = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_l = cv2.bitwise_not(mask_l)

    mask_inv = cv2.bitwise_not(cv2.add(mask_r, mask_l))

    
                        
    # merge live_image and tweak image by mask
    
    eye_r_region = cv2.bitwise_and(tw_img_r_transf, tw_img_r_transf, mask = mask_r)
    eye_l_region = cv2.bitwise_and(tw_img_l_transf, tw_img_l_transf, mask = mask_l)
    tweaked_eyes = cv2.add(eye_r_region, eye_l_region)
    
   
    live_eye_r = cv2.bitwise_and(live_img, live_img, mask = mask_r)
    live_eye_l = cv2.bitwise_and(live_img, live_img, mask = mask_l)
    live_eyes  = cv2.add(live_eye_r, live_eye_l)
    
    if np.isclose(brightness(tweaked_eyes), brightness(live_eyes), rtol=gv.BTOL): # or random.choice([True, False]):
        
    
        face_region = cv2.bitwise_and(live_img, live_img, mask = mask_inv)
        tweaked_img = cv2.add(tweaked_eyes, face_region)
        
        return tweaked_img, True
    else:
        return live_img, False
