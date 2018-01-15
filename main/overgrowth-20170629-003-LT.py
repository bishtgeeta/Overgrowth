import os, sys
import h5py
import cv2
import numpy
import gc
import mpi4py
import matplotlib.pyplot as plt
from scipy import ndimage
from mpi4py import MPI
from time import time
from tqdm import tqdm

sys.path.append(os.path.abspath('../myFunctions'))
import fileIO
import imageProcess
import myCythonFunc
import dataViewer
import misc
import tracking

inputFile = r'W:\geeta\Overgrowth\Low Temperature\20170629-003\20170629-003.avi'
outputFile = r'W:\geeta\Overgrowth\Low Temperature\20170629-003\20170629-003.h5'
inputDir = r'W:\geeta\Overgrowth\Low Temperature\20170629-003'
outputDir = r'W:\geeta\Overgrowth\Low Temperature\20170629-003\output'
pixInNM = 0.53283203
fps = 10
microscope = 'JOEL2010' #'JOEL2010','T12'
camera = 'One-view' #'Orius', 'One-view'
owner = 'Shu Fen'
zfillVal = 6
fontScale = 1
structure = [[1,1,1],[1,1,1],[1,1,1]]


#########################################
# CHANGING PIX IN NM
#########################################

#fp = h5py.File(outputFile, 'r+')
#fp.attrs['pixInNM'] = pixInNM
#fp.attrs['pixInAngstrom'] = pixInNM*10
#fp.close()


#######################################################################
# INITIALIZATION FOR THE MPI ENVIRONMENT
#######################################################################
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#######################################################################

if (rank==0):
    tic = time()
#######################################################################
# DATA PROCESSSING
# 1. READ THE INPUT FILES AND STORE THEM FRAME-WISE IN H5 FILE
# 2. PERFORM BACKGROUND SUBTRACTION (IF REQUIRED)
#######################################################################
#########
# PART 1
##########
#if (rank==0):
    #fp = fileIO.createH5(outputFile)
    #[gImgRawStack,row,col,numFrames] = fileIO.readAVI(inputFile)
    #frameList = range(1,numFrames+1)
    #for frame in frameList:
        #fileIO.writeH5Dataset(fp,'/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal),gImgRawStack[:,:,frame-1])
    
    #fp.attrs['inputFile'] = inputFile
    #fp.attrs['outputFile'] = outputFile
    #fp.attrs['inputDir'] = inputDir
    #fp.attrs['outputDir'] = outputDir
    #fp.attrs['pixInNM'] = pixInNM
    #fp.attrs['pixInAngstrom'] = pixInNM*10
    #fp.attrs['fps'] = fps
    #fp.attrs['microscope'] = microscope
    #fp.attrs['camera'] = camera
    #fp.attrs['owner'] = owner
    #fp.attrs['row'] = row
    #fp.attrs['col'] = col
    #fp.attrs['numFrames'] = numFrames
    #fp.attrs['frameList'] = range(1,numFrames+1)
    #fp.attrs['zfillVal'] = zfillVal
    
    #fileIO.mkdirs(outputDir)
    #fileIO.saveImageSequence(gImgRawStack,outputDir+'/dataProcessing/gImgRawStack')
    
    #del gImgRawStack
    #fp.flush(), fp.close()
    #gc.collect()
#comm.Barrier()

#########
# PART 2
##########
#if (rank==0):
    #print "Inverting the image and performing background subtraction"
#invertFlag=True
#bgSubFlag= True; bgSubSigmaTHT=2; radiusTHT=30
#blurFlag=True; sigma=2

#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#procFrameList = numpy.array_split(frameList,size)

#for frame in tqdm(procFrameList[rank]):
    #gImgProc = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    #if (invertFlag==True):
        #gImgProc = imageProcess.invertImage(gImgProc)
    #if (bgSubFlag==True):
        #gImgProc = imageProcess.subtractBackground(gImgProc, sigma=bgSubSigmaTHT, radius=radiusTHT)
    #if (blurFlag==True):
        #gImgProc = ndimage.gaussian_filter(gImgProc, sigma=sigma)
        #gImgProc = imageProcess.normalize(gImgProc)
    #cv2.imwrite(outputDir+'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)+'.png',gImgProc)

#comm.Barrier()
    
#if (rank==0):
    #for frame in frameList:
        #gImgProc = cv2.imread(outputDir+'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)+'.png',0)
        #fileIO.writeH5Dataset(fp,'/dataProcessing/processedStack/'+str(frame).zfill(zfillVal),gImgProc)
        
#fp.flush(), fp.close()
#comm.Barrier()
########################################################################


#######################################################################
# IMAGE SEGMENTATION
#######################################################################
#if (rank==0):
    #print "Performing segmentation for all the frames"
    
#fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#procFrameList = numpy.array_split(frameList,size)

#areaRange = numpy.array([1500,10000], dtype='float64')

#for frame in tqdm(procFrameList[rank]):
    #gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    #gImgNorm = imageProcess.normalize(gImgRaw,min=0,max=230)
    #gImgProc = fp['/dataProcessing/processedStack/'+str(frame).zfill(zfillVal)].value
    ##bImg = gImgProc>=imageProcess.otsuThreshold(gImgProc)
    #bImg = gImgProc>=myCythonFunc.threshold_kapur(gImgProc.flatten())
    ##bImg = imageProcess.binary_erosion(bImg, iterations=15)
    ##bImg = imageProcess.binary_dilation(bImg, iterations=15)
    #bImg = myCythonFunc.areaThreshold(bImg.astype('uint8'), areaRange=areaRange)
    #bImg = imageProcess.binary_closing(bImg, iterations=6)
    
    ##bImg = imageProcess.convexHull(bImg)
    #bImgBdry = imageProcess.normalize(imageProcess.boundary(bImg))
    #finalImage = numpy.column_stack((numpy.maximum(gImgNorm,bImgBdry), gImgNorm))
    #cv2.imwrite(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.png', finalImage)
#fp.flush(), fp.close()
#comm.Barrier()
#######################################################################


#######################################################################
# CREATE BINARY IMAGES INTO HDF5 FILE
#######################################################################
if (rank==0):
    print "Creating binary images and writing into h5 file"
    
if (rank==0):
    fp = h5py.File(outputFile, 'r+')
else:
    fp = h5py.File(outputFile, 'r')
[row,col,numFrames,frameList] = misc.getVitals(fp)
procFrameList = numpy.array_split(frameList,size)

for frame in tqdm(procFrameList[rank]):
    bImg = cv2.imread(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.png',0)[0:row,0:col]
    bImg = bImg==255
    bImg = imageProcess.fillHoles(bImg)
    bImg = imageProcess.binary_opening(bImg, iterations=1)
    numpy.save(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy', bImg)
   
comm.barrier()
if (rank==0):
    for frame in frameList:
        bImg = numpy.load(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy')
        fileIO.writeH5Dataset(fp,'/segmentation/bImgStack/'+str(frame).zfill(zfillVal),bImg)
        fileIO.delete(outputDir+'/segmentation/result/'+str(frame).zfill(zfillVal)+'.npy')
        
fp.flush(), fp.close()
comm.Barrier()
######################################################################

#######################################################################
# LABELLING PARTICLES
#######################################################################
centerDispRange = [50,50]
perAreaChangeRange = [80,80]
missFramesTh = 10
    
if (rank==0):
    print "Labelling segmented particles"
    fp = h5py.File(outputFile, 'r+')
    [row,col,numFrames,frameList] = misc.getVitals(fp)
    maxID, occurenceFrameList = tracking.labelParticles(fp, centerDispRange=centerDispRange, perAreaChangeRange=perAreaChangeRange, missFramesTh=missFramesTh, structure=structure)
    fp.attrs['particleList'] = range(1,maxID+1)
    numpy.savetxt(outputDir+'/frameOccurenceList.dat',numpy.column_stack((fp.attrs['particleList'],occurenceFrameList)),fmt='%d')
    fp.flush(), fp.close()
comm.Barrier()

if (rank==0):
    print "Generating images with labelled particles"
fp = h5py.File(outputFile, 'r')
tracking.generateLabelImages(fp,outputDir+'/segmentation/tracking')
fp.flush(), fp.close()
comm.Barrier()
#######################################################################


#######################################################################
# REMOVING UNWANTED PARTICLES
########################################################################
#keepList = [1]
#removeList = []

#if (rank==0):
	#print "Removing unwanted particles"

#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#particleList = fp.attrs['particleList']
#if not removeList:
	#removeList = [s for s in particleList if s not in keepList]
#tracking.removeParticles(fp,removeList,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################


#######################################################################
# GLOBAL RELABELING OF PARTICLES
#######################################################################
#correctionList = [[3,4],[2,3],[1,2]]

#if (rank==0):
	#print "Global relabeling of  particles"
    
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#tracking.globalRelabelParticles(fp,correctionList,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################


#######################################################################
# FRAME-WISE CORRECTION OF PARTICLE LABELS
#######################################################################
#frameWiseCorrectionList = [\
#[range(455,459),[4,2]],\
#[[582],[5,2]],\
#[range(589,591),[5,2]],\
#[[715],[3,1]],\
#[range(795,797),[6,2]],\
#[range(800,802),[7,2]],\
#[range(803,814),[6,2]]\
#]

#if (rank==0):
    #print "Frame-wise relabeling of  particles"
    
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#tracking.framewiseRelabelParticles(fp,frameWiseCorrectionList,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################


#######################################################################
# RELABEL PARTICLES IN THE ORDER OF OCCURENCE
########################################################################
#if (rank==0):
    #fp = h5py.File(outputFile, 'r+')
#else:
    #fp = h5py.File(outputFile, 'r')
#tracking.relabelParticles(fp,comm,size,rank)
#fp.flush(), fp.close()    
#comm.Barrier()
#######################################################################

#######################################################################
# GENERATING IMAGES WITH LABELLED PARTICLES
########################################################################
#if (rank==0):
    #print "Generating images with labelled particles"
#fp = h5py.File(outputFile, 'r')
#tracking.generateLabelImages(fp,outputDir+'/segmentation/tracking',fontScale,size,rank)
#fp.flush(), fp.close()
#comm.Barrier()
######################################################################


#######################################################################
# FINDING OUT THE MEASURES FOR TRACKED PARTICLES
########################################################################
#if (rank==0):
    #print "Finding measures for tracked particles"

#fp = h5py.File(outputFile, 'r')
#[row,col,numFrames,frameList] = misc.getVitals(fp)
#particleList = fp.attrs['particleList']
#zfillVal = fp.attrs['zfillVal']
#procFrameList = numpy.array_split(frameList,size)
#fps = fp.attrs['fps']
#pixInNM = fp.attrs['pixInNM']

#outFile = open(str(rank)+'.dat','wb')

##particleList = [1,2]

#area=True
#perimeter=True
#circularity=False
#pixelList=False
#bdryPixelList=False
#centroid=True
#intensityList=False
#sumIntensity=False
#effRadius=False
#radius=False
#circumRadius=False
#inRadius=False
#radiusOFgyration=False
#orientation=True

#for frame in procFrameList[rank]:
    #labelImg = fp['/segmentation/labelStack/'+str(frame).zfill(zfillVal)].value
    #gImgRaw = fp['/dataProcessing/gImgRawStack/'+str(frame).zfill(zfillVal)].value
    #outFile.write("%f " %(1.0*frame/fps))
    #for particle in particleList:
        #bImg = labelImg==particle
        #if (bImg.max() == True):
            #label, numLabel, dictionary = imageProcess.regionProps(bImg, gImgRaw, structure=structure, centroid=centroid, area=area, perimeter=perimeter,orientation=orientation)
            #outFile.write("%f %f %f %f %f " %(dictionary['centroid'][0][1]*pixInNM, (row-dictionary['centroid'][0][0])*pixInNM, dictionary['area'][0]*pixInNM*pixInNM, dictionary['perimeter'][0]*pixInNM, dictionary['orientation'][0]))
        #else:
            #outFile.write("nan nan nan nan nan ")
    #outFile.write("\n")
#outFile.close()
#fp.flush(), fp.close()
#comm.Barrier()

#if (rank==0):
    #for r in range(size):
        #if (r==0):
            #measures = numpy.loadtxt(str(r)+'.dat')
        #else:
            #measures = numpy.row_stack((measures,numpy.loadtxt(str(r)+'.dat')))
        #fileIO.delete(str(r)+'.dat')
    #measures = measures[numpy.argsort(measures[:,0])]
    #numpy.savetxt(outputDir+'/imgDataNM.dat', measures, fmt='%.6f')
#######################################################################


#######################################################################
# FINDING OUT RELATIVE DISTANCE AND ANGLE BETWEEN PARTICLES
##############################################################################
#if (rank==0):
    #print "Finding the relative distance and angle"
    #txtfile = numpy.loadtxt(outputDir+'/imgDataNM.dat')
    #time = txtfile[:,0]
    #x1 = txtfile[:,1]
    #y1 = txtfile[:,2]
    #x2 = txtfile[:,6]
    #y2 = txtfile[:,7]
    #x3 = txtfile[:,11]
    #y3 = txtfile[:,12]
    #see_3rd_particle = numpy.isnan(x3)
    #relative_distance = numpy.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    #relative_distance[~see_3rd_particle] = 0
    #slopes1 = txtfile[:,5]
    #slopes2 = txtfile[:,10]
    #intersection_angles = numpy.empty(x1.shape)
    #intersection_angles[:] = numpy.NaN
    #for frame_num in range(x1.shape[0]):
        #c1 = [x1[frame_num], y1[frame_num]]
        #c2 = [x2[frame_num], y2[frame_num]]
        #m1 = slopes1[frame_num]
        #m2 = slopes2[frame_num]
        #intersection_angles[frame_num] = imageProcess.get_intersection_angle(c1, m1, c2, m2)
		
    #numpy.savetxt(outputDir+'/relative_distance.dat', numpy.column_stack([time, x1, y1, x2, y2, relative_distance, slopes1, slopes2, intersection_angles]),fmt='%.6f')

    
#######################################################################
# Plotting Graph Between Time and Relative Distance\Angles
#######################################################################

#def remove_nan_for_plot(x,y):
    #not_nan = ~numpy.isnan(y)
    #return x[not_nan], y[not_nan]


#def plot_line(x, y, xlabel, ylabel, figsize=[4,2.5], 
					#xlimits=None, ylimits=None, savefile=None):

    #x, y = remove_nan_for_plot(x, y)
    #plt.figure(figsize=figsize)
    #plt.plot(x, y, '-o', color='steelblue', lw=1, mfc='none', mec='orangered', ms=2)
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)
    #if xlimits is not None:
        #plt.xlim(xlimits)
    #if ylimits is not None:
        #plt.ylim(ylimits)
    #plt.tight_layout()
    #if savefile is not None:
        #plt.savefig(savefile, dpi=300)
    #plt.show()

#if (rank == 0):
    #txtfile = numpy.loadtxt(outputDir+'/relative_distance.dat')
    #time = txtfile[:,0]
    #relative_distance = txtfile[:,5]
    #slope_difference = txtfile[:,8]
    #plot_line(x=time, y=relative_distance, xlabel='time (seconds)', ylabel='relative distance (nm)', 
				#xlimits=[0,20], ylimits=[-20,300], 
				#savefile=outputDir+'/nanorod_relative_distance.png')
    #plot_line(x=time, y=slope_difference,xlabel='time (seconds)', ylabel='slope_difference (degrees)', 
				#xlimits=[0,20], ylimits=[-200, 200], 
				#savefile=outputDir+'/rod_slope_difference.png')
