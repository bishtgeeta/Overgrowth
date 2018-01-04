import numpy
import pims
from libtiff import TIFF
from mpi4py import MPI

####################################################
# USER INPUTS
####################################################
inputFileList = [\
'/mnt/komodo-images/seewee/Extract/13-58-38.348.seq',\
'/mnt/komodo-images/seewee/Extract/13-58-38.348.seq'\
]
outputDirList = [\
'/mnt/NAS-Drive/Utkarsh-Share/SeeWee/seqTest1',\
'/mnt/NAS-Drive/Utkarsh-Share/SeeWee/seqTest2'\
]
####################################################


####################################################
# INITIALIZING THE MPI ENVIRONMENT
####################################################
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
####################################################


####################################################
# READING THE INPUT FILES AND STORING AS TIF SEQUENCE
####################################################
for inputFile,outputDir in zip(inputFileList,outputDirList):
    if (rank==0):
        print inputFile
    images = pims.open(inputFile)
    [row,col,numFrames] = images.header_dict['height'],images.header_dict['width'],images.header_dict['allocated_frames']
    frameList = range(1,numFrames+1)
    procFrameList = numpy.array_split(frameList,size)
    
    for frame in procFrameList[rank]:
        gImg = images[frame-1]
        outputFile = outputDir+'/'+str(frame).zfill(6)+'.tif'
        outTif = TIFF.open(outputFile, mode='w')
        outTif.write_image(gImg)
        del outTif,gImg
    comm.Barrier()
####################################################
