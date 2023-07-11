import os, glob
import numpy as np
import math

########################
def getLatestModelPath(trainingPath="training/", pattern="phi", filetype="json"):
    directories = glob.glob(trainingPath+"/*"+pattern+"*."+filetype)
    directories.sort(key=os.path.getmtime)
    return directories[-1][len(trainingPath):]

########################
def phi_mpi_pi(x):
    '''Returns phi angle in the interval [-PI,PI)'''
    while x >= math.pi: x -= 2*math.pi
    while x < -math.pi: x += 2*math.pi

    return x

