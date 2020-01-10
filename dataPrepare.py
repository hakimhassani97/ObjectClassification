import shutil
import os
from dataStructure import classes
import glob

sourceDataPathRGB='fullTracks/RGB_fullTracks'
sourceDataPathBinary='fullTracks/binary_fullTracks'

dataPath='data/'
dataPathRGB=dataPath+'dataRGB'
dataPathBinary=dataPath+'dataBinary'
# classes=['bike','boat','canoe','car','human','noise','pickup','truck','van']
# classesCodes=['1','2','3','4','5','6','7','8','9']

def createClassDirectories(destDataPath):
    # creates a directory for each class
    for c,value in classes.items():
        dirName=destDataPath+'/'+c
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        else:
            print(dirName+' already exists')

def copyData(sourceDataPath,destDataPath):
    # copy images of each class
    for c,value in classes.items():
        dirName=destDataPath+'/'+c
        if os.path.exists(dirName):
            for info in value['info']:
                videoName=info['videoName']
                trajectoryName=info['trajectoryName']
                trajectoryName=str(trajectoryName)
                sourcePath=sourceDataPath+'/'+videoName+'/'+trajectoryName+'/'
                destPath=dirName+'/'
                files=os.listdir(sourcePath)
                for file in files:
                    shutil.copy2(sourcePath+file,destPath+videoName+'_'+trajectoryName+'_'+file)
    return
def dataPrep(sourceDataPath,destDataPath):
    # create dataPath folder if not existing
    if not os.path.exists(dataPath):
        os.mkdir(dataPath)
    # create destDataPath folder inside dataPath folder if not existing
    if not os.path.exists(destDataPath):
        os.mkdir(destDataPath)
    # create a directory for each class
    createClassDirectories(destDataPath)
    # copy images of each class to its directory
    copyData(sourceDataPath,destDataPath)

def main():
    # prepare RGB data
    dataPrep(sourceDataPathRGB,dataPathRGB)
    # prepare Binary data
    dataPrep(sourceDataPathBinary,dataPathBinary)


# main program
main()
