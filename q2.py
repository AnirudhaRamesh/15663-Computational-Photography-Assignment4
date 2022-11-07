#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from scipy.interpolate import interp2d
from src.cp_hw2 import *
from tqdm import tqdm


# In[4]:


# read video and select frames
subsample_skip = 4
vc = cv2.VideoCapture('data-dump/DSC_1306.MOV')
frames = []
success, frame = vc.read()
frames.append(frame[::subsample_skip,::subsample_skip,::-1])
it = 0
while True : 
    success, frame = vc.read()
    if success is False : break
    if it % 3 == 0: frames.append(frame[::subsample_skip,::subsample_skip,::-1])
    it += 1
frames = np.array(frames)
frames = frames.transpose(1,2,3,0)


# In[5]:


# Get luminance channel for template matching
@np.vectorize
def linearize_image(C_nonlinear) : 
    
    if C_nonlinear <= 0.0404482 : 
        return C_nonlinear / 12.92
    else : 
        out_num = ( C_nonlinear + 0.055 ) ** 2.4
        out_den = 1.055 ** 2.4
        
        return out_num / out_den
    
def get_luminance(I) : 
    print(I.shape)
    if len(I.shape) == 3 : I = np.expand_dims(I,3)
    I_linear = linearize_image(I)
    I_XYZ = np.zeros(I_linear.shape)
    for foc_im in range(I_linear.shape[-1]) : 
        I_XYZ[:,:,:,foc_im] = lRGB2XYZ(I_linear[:,:,:,foc_im])
    I_luminance = I_XYZ[:,:,1,:] # take just Y channel for relative luminance 
    return I_luminance


# In[20]:


# template selection in the middle frame of video

# DSC 1298
# pink pig subsample frame_id % 10 == 0 (31 frames)
# bbox= (np.array([340,130,440,230])// ( subsample_skip / 2 )).astype('int') # [[row1,col1,row2,col2]] sp, ep = (bbox[1],bbox[0]) , (bbox[3], bbox[2])

# pink pig subsample frame_id % 1 == 0 (297 frames) or %2 (149 frames)
# bbox= (np.array([330,135,430,235])// ( subsample_skip / 2 )).astype('int') # [[row1,col1,row2,col2]] sp, ep = (bbox[1],bbox[0]) , (bbox[3], bbox[2])

# DSC 1306
# pink pig subsample frame_id % 3 == 0 (102 frames)
# bbox= (np.array([360,145,460,245])// ( subsample_skip / 2 )).astype('int') # [[row1,col1,row2,col2]] sp, ep = (bbox[1],bbox[0]) , (bbox[3], bbox[2])

# bag of banana chips in the background (102 frames, subsample by % 3)
# bbox= (np.array([230,465,310,545])// ( subsample_skip / 2 )).astype('int') # [[row1,col1,row2,col2]] sp, ep = (bbox[1],bbox[0]) , (bbox[3], bbox[2])

# coffee mug subsample frame_id % 3 == 0 (102 frames)
bbox= (np.array([340,555,410,635])// ( subsample_skip / 2 )).astype('int') # [[row1,col1,row2,col2]] sp, ep = (bbox[1],bbox[0]) , (bbox[3], bbox[2])


sp, ep = (bbox[1],bbox[0]) , (bbox[3], bbox[2])
mid_im = frames[:,:,:,frames.shape[3]//2].copy()
mid_im_disp = mid_im.copy()
mid_im_disp = cv2.rectangle(mid_im_disp,sp,ep,(255,0,0),2)
plt.imshow(mid_im_disp)
plt.title('template selected')
plt.show()

template = mid_im[bbox[0] :  bbox[2] , bbox[1] : bbox[3]]
plt.imshow(template)
plt.title('template')
plt.show()

# template = np.mean(template,2)
# mid_im = np.mean(mid_im,2)
template = get_luminance(template)
mid_im = get_luminance(mid_im)


# In[21]:


# Testing to see if our normxcorr function works

def get_normxcorr(img, template, bbox) :
    d1 = np.sum((template - template.mean())**2)
    normxcorr = np.zeros(img.shape)

    bbox_center = [(bbox[0] + bbox[2])/2, (bbox[1]+bbox[3])/2]
    bbox_size = np.array(template.shape) * 2
    count = 0

    for i in range(img.shape[0]) : 

        if np.abs(i - bbox_center[0]) > bbox_size[0] : continue

        for j in range(img.shape[1]) :

            if np.abs(j - bbox_center[1]) > bbox_size[1] : continue

            if img[i:i+template.shape[0], j:j+template.shape[1]].shape != template.shape : continue

            count += 1
            num = np.sum(img[i:i+template.shape[0], j:j+template.shape[1]] * template)
            I_box = img[i:i+template.shape[0], j:j+template.shape[1]] / img[i:i+template.shape[0], j:j+template.shape[1]].sum()
            d2 = np.sum((img[i:i+template.shape[0], j:j+template.shape[1]] - I_box) ** 2)
            normxcorr[i,j] = num / np.sqrt(d1*d2)
            
    return normxcorr

def calculate_shift(normxcorr, bbox) : 
    
    y, x = np.unravel_index(np.argmax(normxcorr), normxcorr.shape)
    
    y_mid_im, x_mid_im = bbox[0], bbox[1]
    
    shift_x = x - x_mid_im # col shift
    shift_y = y - y_mid_im # row shift

    return shift_x, shift_y

def display_correspondance(normxcorr, frame=None) :
    y, x = np.unravel_index(np.argmax(normxcorr), normxcorr.shape)
    plt.imshow(normxcorr**4)
    plt.show()
    plt.imshow(normxcorr**4)
    plt.scatter(x,y,100,'r')
    plt.show()
    if frame is not None :
        plt.imshow(frame)
        plt.scatter(x,y,100,'r')
        plt.show()

normxcorr = get_normxcorr(mid_im[:,:,0], template[:,:,0], bbox)
display_correspondance(normxcorr)

print('giving middle image and template, we should get shift equal to (0,0)')
print('calculated shift is ', calculate_shift(normxcorr, bbox))


# In[22]:


# frame interpolator
L_interp = []
for i in range(frames.shape[3]) :
        temp = []
        temp.append(interp2d(np.arange(frames.shape[1]),np.arange(frames.shape[0]),frames[:,:,0,i]))
        temp.append(interp2d(np.arange(frames.shape[1]),np.arange(frames.shape[0]),frames[:,:,1,i]))
        temp.append(interp2d(np.arange(frames.shape[1]),np.arange(frames.shape[0]),frames[:,:,2,i]))
        L_interp.append(temp)


# In[23]:


shifted_frames = np.zeros(frames.shape)
shift_array = []
frames_luminance = get_luminance(frames)
combined_frame = np.zeros(frames[:,:,:,0].shape)

# calculate shift
print('calculating shift')
for fi in tqdm(range(frames.shape[3])) : 
    normxcorr = get_normxcorr(frames_luminance[:,:,fi], template[:,:,0], bbox)
    display_correspondance(normxcorr,frames[:,:,:,fi])
    shift_array.append(calculate_shift(normxcorr, bbox))

print('shifting frames')
# shift frames
for fi in tqdm(range(frames.shape[3])) :
    temp = np.zeros(frames[:,:,:,0].shape)
    cur_fun = L_interp[fi]
    temp[:,:,0] = cur_fun[0](np.arange(frames.shape[1])+shift_array[fi][0], np.arange(frames.shape[0])+shift_array[fi][1])
    temp[:,:,1] = cur_fun[1](np.arange(frames.shape[1])+shift_array[fi][0], np.arange(frames.shape[0])+shift_array[fi][1])
    temp[:,:,2] = cur_fun[2](np.arange(frames.shape[1])+shift_array[fi][0], np.arange(frames.shape[0])+shift_array[fi][1])
    shifted_frames[:,:,:,fi] = temp.copy() 
    
    combined_frame += temp.copy() / frames.shape[3]


# In[24]:


plt.imshow(shifted_frames[...,19].astype('int'))


# In[25]:


plt.imshow(combined_frame.astype('int'))
cv2.imwrite('output-dump/rename-me.png',combined_frame[...,::-1])


# In[ ]:


shift_array


# In[16]:


frames.shape


# In[ ]:


bbox


# In[ ]:




