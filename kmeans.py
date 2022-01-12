# import os 


# im_path = '/Users/kevin/Desktop/Heart_Segmentation/data'


# name_path_dir = {
#     'R1-1-' : 'R1/i1',
#     'R1-2-' : 'R1/i2',
#     'R2-2-' : 'R2/i2',
#     'R3-2-' : 'R3/i1',
#     'R3-2-' : 'R2/i2',
#     'r4a1-' : 'R4/a1',
#     'r4a1-' : 'R4/a2',
#     'r4b1-' : 'R4/b1',
#     'r4b2-' : 'R4/b2',
#     'r5a1-' : 'R5/a1',
# }


# def create_video():
#     for p in name_path_dir.items():
#         inp = os.path.join(im_path, 'images', p[1], p[0]+'%d.jpg')
#         outp = os.path.join(im_path, p[0][:-1] + '.mp4')
#         os.system(f'ffmpeg -r 30 -i {inp} -c:v libx264 -vf fps=30 -pix_fmt yuv420p {outp}')
        

# create_video()
import scipy.io as sio
import os
import math
import numpy as np
import pandas as pd
import io
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def gray2rgb(image):
    width, height = image.shape
    out = np.empty((width, height, 3), dtype=np.uint8)
    out[:, :, 0] = image
    out[:, :, 1] = image
    out[:, :, 2] = image
    return out

def plot_result(img:np.ndarray, mask:np.ndarray, fig_path:str, name_panel:str, name_single:str) -> None:
#     img=mpimg.imread(im_path)
    fig = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    
    ax1 = plt.subplot(2, 2, 1)
    imgplot = plt.imshow(img, cmap="gray")


    ax2 = plt.subplot(2, 2, 2)
    imgplot = plt.imshow(mask, cmap="gray")


    ax3 = plt.subplot(2, 2, 3)
    img = np.multiply(img,mask)
    imgplot = plt.imshow(img, cmap="gray")


    ax4 = plt.subplot(2, 2, 4)
    rgb = gray2rgb(img)

    for i in range(0, 100):
        for j in range(0, 100):
            if(rgb[i][j][0] == 0):
                rgb[i][j][0] = 255
    imgplot = plt.imshow(rgb)
    
    # save single red mask image
    if name_single != '':
        Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(fig_path, name_single))
    
    ax1.title.set_text('Original Image')
    ax2.title.set_text('Mask Image')
    ax3.title.set_text('Masked Original Heart Image')
    ax4.title.set_text('Colored Masked Heart Image')
    
    # save the whole panel
    if name_panel != '':
        plt.savefig(os.path.join(fig_path, name_panel))
#     plt.show()

    plt.clf()
    
def k_means(data, loop=50):
    _,_,dim = data.shape
    c1 = np.ones([dim,1])
    c2 = np.zeros([dim,1])

    ss = np.zeros([100,100])
    oldss = ss

    for i in range(loop):
    #     print(i)
        c1 = np.array([np.repeat(c1[i], 10000).reshape(100,100) for i in range(dim)])
        c1 = np.moveaxis(c1, 0, -1)

        c2 = np.array([np.repeat(c2[i], 10000).reshape(100,100) for i in range(dim)])
        c2 = np.moveaxis(c2, 0, -1)

        d1 = np.sum((data-c1)**2, 2)
        d2 = np.sum((data-c2)**2, 2)

        ss[d1<d2] = 1
        ss[d1>d2] = 2

        if len(ss[ss==1]) > 0:
            tmp = data
            xxx = np.ones([100,100])
            xxx[ss==2] = 0
            tmp = tmp * np.repeat(xxx[:, :, np.newaxis], dim, axis=2)
            c1 = np.sum(tmp, (1,0)) / len(ss[ss==1])
            c1 = c1.reshape(dim,1)

        else:
            c1 = np.random.rand(dim,1)


        if len(ss[ss==2]) > 0:
            tmp = data
            xxx = np.ones([100,100])
            xxx[ss==1] = 0
            tmp = tmp * np.repeat(xxx[:, :, np.newaxis], dim, axis=2)
            c2 = np.sum(tmp, (1,0)) / len(ss[ss==2])
            c2 = c2.reshape(dim,1)
        else:
            c2 = np.random.rand(dim,1)
    return ss

def over_thresh(mask, r:int, i, j, thresh):
    '''Help function for remove_noise function.
    '''
    count = -1
    v = mask[i][j]
    for m in range(i-r, i+r+1):
        for n in range(j-r, j+r+1):
            if mask[m][n] == v:
                count+=1
                if count >= thresh:
                    return True
    return False 

def remove_noise(mask:np.ndarray, thresh = 1, exten = 1, rheart=True, rnonheart=True):
    '''To remove noises from the mask.
    '''
    s1 = mask.shape[0]
    s2 = mask.shape[1]
    t = 0
    for i in tqdm(range(s1)):
        for j in range(s2):
            if (not rheart) and mask[i][j] == 1:
                continue
            if (not rnonheart) and mask[i][j] == 0:
                continue
            if (i < exten) or (i >= s1-exten) or (j < exten) or (j >= s2-exten):
                continue 
            if not over_thresh(mask, exten, i, j, thresh):
                t+=1
                mask[i][j] = 1-mask[i][j]
    print(t)
    return mask

import copy
import argparse
def argparser():
    parser = argparse.ArgumentParser(description="K-Means.")
    parser.add_argument("--n", required=True,help="Name")
    parser.add_argument("--f", required=True,help="Frame")
    return parser.parse_args()

args = argparser()
name = args.n
fr = int(args.f)

print(name + ' with ' + str(fr) + ' frame')
dt = sio.loadmat('../data/' + name + '.mat')
data = dt['images1'].astype(np.double)

img = data[:,:,0]
img = (img - np.min(img)) / (np.max(img) - np.min(img)) *255 
# Image.fromarray((img).astype(np.uint8)).save(os.path.join('../plots', name+'origin.jpg')) # save single original

dmin = np.repeat(np.min(data, 2)[:, :, np.newaxis], 1024, axis=2)
dmax = np.repeat(np.max(data, 2)[:, :, np.newaxis], 1024, axis=2)
data = (data - dmin) / (dmax - dmin) # minmax normalization


data = data[:,:,:fr] # slice frame 
print(data.shape)
ss = k_means(data, 50)

# if mask is reversed, change here to ss = np.where(ss == 2, 0, 1)
ss = np.where(ss == 2, 1, 0)

print(f'Heart Ratio: {str((1 - np.count_nonzero(ss) / 10000))}')
plot_result(img, ss, '../outputs/',  name+"KMeansOrigin"+str(fr), name+'kmeans'+str(fr)+'_orginal.png')

from scipy.ndimage import median_filter
import cv2
import numpy as np
from matplotlib import pyplot as plt

mf = copy.deepcopy(ss)
p  = median_filter(mf ,(3,3)) # [[1,0,0],[1,0,1]]

tmp = mf + p 
tmp = np.where(tmp==1, 1, 0)
print(np.count_nonzero(tmp))

# if mask is reversed, change here to ss = np.where(ss == 2, 0, 1)
plot_result(img, p, '../outputs/',  name+"KMeansMedian"+str(fr), name+'kmeans'+str(fr)+'_median.png')


kk = copy.deepcopy(ss)
kk = remove_noise(kk, 20, 3)
kk = remove_noise(kk, 10, 2)
kk = remove_noise(kk, 3, 1)
tmp = kk + ss 
tmp = np.where(tmp==1, 1, 0)
print(np.count_nonzero(tmp))

plot_result(img, kk, '../outputs/', name+"KMeansPostp"+str(fr), name+'kmeans' + str(fr) +'_postprocess.png')



# def slice_prdict(data, start, end, name_single):
    
#     d = data[:,:, start:end]
#     ss = k_means(d, 50)
#     # if mask is reversed, change here to ss = np.where(ss == 2, 0, 1)
#     ss = np.where(ss == 2, 1, 0)
#     if(np.count_nonzero(ss) < 5000):
#         ss = np.where(ss == 1, 0, 1)
#     print(f'Heart Ratio: {str((np.count_nonzero(ss) / 10000))}')
#     plot_result(img, ss, '../outputs/',  '', name_single)

    
# slice_prdict(data, 0, 30, '1kmeans_orginal.png')
# slice_prdict(data, 30, 60, '2kmeans_orginal.png')
# slice_prdict(data, 60, 90, '3kmeans_orginal.png')
# slice_prdict(data, 90, 120, '4kmeans_orginal.png')
# slice_prdict(data, 120, 150, '5kmeans_orginal.png')
