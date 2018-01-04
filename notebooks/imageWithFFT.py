import numpy
import cv2
import imageio
import os
from libtiff import TIFF
from mpi4py import MPI
import matplotlib.pyplot as plt

def mkdir(dirName):
    if (os.path.exists(dirName) == False):
        os.makedirs(dirName)
        
def normalize(gImg, min=0, max=255):
    if (gImg.max() > gImg.min()):
        gImg = 1.0*(max-min)*(gImg - gImg.min())/(gImg.max() - gImg.min())
        gImg=gImg+min
    elif (gImg.max() > 0):
        gImg[:] = max
    gImg=gImg.astype('uint8')
    return gImg
    
inputDir = '/mnt/NAS-Drive/SeeWee-Share/20161114/17_15-13-28.362_Export'
outputDir = '/mnt/NAS-Drive/Utkarsh-Share/SeeWee/Nucleation/20161114/17_15-13-28.362_Export'
header = '17_15-13-28.362_'
footer = '.tif'
frameList = range(8793)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
procFrameList = numpy.array_split(frameList,size)

for frame in procFrameList[rank]:
    inputFile = inputDir+'/'+header+str(frame)+footer
    outputFile = outputDir+'/'+header+str(frame)+footer
    inTif = TIFF.open(inputFile, mode='r')
    outTif = TIFF.open(outputFile, mode='w')
    gImgRaw = inTif.read_image()
    fImg = numpy.fft.fftshift(numpy.fft.fftn(gImgRaw))
    fImgAbs = numpy.abs(fImg)**2
    fImgLog=normalize(numpy.log10(fImgAbs)+1)
    finalImg = numpy.column_stack((gImgRaw,fImgLog))
    outTif.write_image(finalImg)
del inTif,outTif

#inputFile = r'Z:\utkarsh\ForErmanno\38.avi'

#reader = imageio.get_reader(inputFile)
#col, row = reader.get_meta_data()['size']
#numFrames = reader.get_meta_data()['nframes']
#frameList=range(1,numFrames+1)
#mkdir('FFT')

#for i, img in enumerate(reader):
    #gImgRaw = img[:,:,0]
    #fImg = numpy.fft.fftshift(numpy.fft.fftn(gImgRaw))
    #fImgAbs = numpy.abs(fImg)**2
    #fImgLog=normalize(numpy.log10(fImgAbs)+1)
    #finalImg = numpy.column_stack((gImgRaw,fImgLog))
    #cv2.imwrite('FFT/'+str(i+1)+'.png', finalImg)
