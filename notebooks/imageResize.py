import numpy
import cv2
from libtiff import TIFF
from mpi4py import MPI

####################################################
# USER INPUTS
####################################################
inputDirList = [\
'/mnt/komodo-images/seewee/Extract/13-53-52.845',\
'/mnt/komodo-images/seewee/Extract/13-58-38.348'\
]
outputDirList = [\
'/mnt/NAS-Drive/Utkarsh-Share/SeeWee/Nucleation/ExtractResize/13-53-52.845',\
'/mnt/NAS-Drive/Utkarsh-Share/SeeWee/Nucleation/ExtractResize/13-58-38.348'\
]
headerList = [\
'frame_',\
'frame_'\
]
footerList = [\
'.tiff',\
'.tiff'\
]
firstFrameList = [\
0,\
0\
]
lastFrameList = [\
16629,\
12483\
]
scaleList = [\
0.25,\
0.25\
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
# READING THE INPUT IMAGE SEQUENCE AND SCALING IT
####################################################
for inputDir,outputDir,header,footer,firstFrame,lastFrame,scale in zip(inputDirList,outputDirList,headerList,footerList,firstFrameList,lastFrameList,scaleList):
    if (rank==0):
        print inputDir
    frameList = range(firstFrame,lastFrame+1)
    procFrameList = numpy.array_split(frameList,size)

    for frame in procFrameList[rank]:
        inputFile = inputDir+'/'+header+str(frame).zfill(4)+footer
        outputFile = outputDir+'/'+header+str(frame).zfill(6)+footer
        if ('png' in footer):
            gImg = cv2.imread(inputFile,0)
        elif ('tif' in footer):
            inTif = TIFF.open(inputFile, mode='r')
            gImg = inTif.read_image()
            del inTif
        [row,col] = gImg.shape
        if (scale<1):
            gImg = cv2.resize(gImg,(int(col*scale),int(row*scale)),interpolation=cv2.INTER_AREA)
        else:
            gImg = cv2.resize(gImg,(int(col*scale),int(row*scale)),interpolation=cv2.INTER_CUBIC)
        outTif = TIFF.open(outputFile, mode='w')
        outTif.write_image(gImg)
        del outTif,gImg
    comm.Barrier()
####################################################
