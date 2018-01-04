from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
from scipy import ndimage
import h5py

zfillVal = 6
frame=100
fp = h5py.File('/mnt/NAS-Drive/Utkarsh-Share/Utkarsh/MonolayerSelfAssembly/20161019-115/20161019-115_driftCorrect-crop.h5', 'r')

gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
gImgProc = fp['/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)].value
gImgRaw = 255-gImgRaw
gImgRaw = ndimage.gaussian_filter(gImgRaw, sigma=1)
blobs_log1 = blob_log(gImgRaw, min_sigma=2, max_sigma=10, num_sigma=50, threshold=.2)
blobs_log1[:, 2] = blobs_log1[:, 2] * sqrt(2)

blobs_log2 = blob_log(gImgProc, min_sigma=2, max_sigma=10, num_sigma=50, threshold=.2)
blobs_log2[:, 2] = blobs_log2[:, 2] * sqrt(2)

blobs_list = [blobs_log1,blobs_log2]
colors = ['yellow','red']
titles = ['Laplacian of Gaussian','Laplacian of Gaussian']
sequence = zip(blobs_list, colors, titles, [1,2])

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
for blobs, color, title, num in sequence:
    ax.set_title(title)
    ax.imshow(gImgRaw)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=0.5, fill=False)
        ax.add_patch(c)
    plt.savefig(str(num)+'.png',format='png')
    plt.cla()
plt.close()
# Compute radii in the 3rd column.
#blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

#blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
#blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

#blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

#blobs_list = [blobs_log, blobs_dog, blobs_doh]
#colors = ['yellow', 'lime', 'red']
#titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          #'Determinant of Hessian']
#sequence = zip(blobs_list, colors, titles)


#fig,axes = plt.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
#axes = axes.ravel()
#for blobs, color, title in sequence:
    #ax = axes[0]
    #axes = axes[1:]
    #ax.set_title(title)
    #ax.imshow(image, interpolation='nearest')
    #for blob in blobs:
        #y, x, r = blob
        #c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        #ax.add_patch(c)

#plt.show()
