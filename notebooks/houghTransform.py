import os,sys
import matplotlib.pyplot as plt
import cv2
import numpy
import h5py
from skimage.transform import (hough_line, hough_line_peaks,probabilistic_hough_line)
import numpy as np

sys.path.append(os.path.abspath('../myFunctions'))
import fileIO
import imageProcess
import myCythonFunc
import dataViewer
import misc
import tracking

outputFile = r'Z:\Geeta-Share\cubes assembly\20160614-001-output\20160614-001.h5'
fp = h5py.File(outputFile, 'r')
labelImg = fp['/segmentation/labelStack/'+str(1).zfill(6)].value

fig = plt.figure(figsize=(3,3))
ax = fig.add_axes([0,0,1,1])

for label in [2]:
    bImg = labelImg==label
    bImgBdry = imageProcess.boundary(bImg)
    
    h, theta, d = hough_line(bImgBdry)
    
    ax.imshow(bImgBdry, cmap=plt.cm.gray)
    rows, cols = bImgBdry.shape
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=4)):
        print numpy.rad2deg(angle)
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
        ax.plot((0, cols), (y0, y1), '-r')
    ax.axis((0, cols, rows, 0))
    #ax.set_title('Detected lines')
    ax.set_axis_off()
    
plt.show()
