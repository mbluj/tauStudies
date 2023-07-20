#!/usr/bin/env python

import os, re, sys
#import commands
import math
import urllib
import glob
import imp

import CRABClient#from CRABAPI.RawCommand import crabCommand
from crab3 import *

import argparse # it needs to come after ROOT import

#########################################
def prepareCrabCfg(eventsPerJob,
                   numberOfJobs,
                   outLFNDirBase,
                   storage_element,
                   outputDatasetTag,
                   inputDataset,
                   tauIdsToRun = [],
                   prefetch = False,
                   whiteList = [],
                   blackList = [],
                   ignoreLocality = False,
                   runLocal = False):

    requestLabel = "TauTreeForPi0Study"
    requestName = inputDataset.split("/")[1]+"_"+requestLabel
    requestName+="_"+outputDatasetTag

    outputDatasetTag = inputDataset.split("/")[1]+"_"+inputDataset.split("/")[2].split("-")[0]+"_"+requestLabel+"_"+outputDatasetTag

    print("requestName:",requestName,          
          "inputDataset:",inputDataset,
          "outputDatasetTag:",outputDatasetTag)

    ##Modify CRAB3 configuration
    config.JobType.allowUndistributedCMSSW = False
    config.JobType.pluginName = 'Analysis'
    inputFiles = ['../python/produceTauTreeForPi0Study.py',
                  '../python/Var.py',
                  '../python/relValTools.py',
                  '../python/tau_ids.py']
    if tauIdsToRun:
        # need to build final cfg here as crab packs only process into picked file
        # hack to set tauIds using args
        cfgName = '../python/runTauIDsOnMiniAOD.py'
        tauIdsStr = 'mvaIds='
        for tauId in tauIdsToRun:
            tauIdsStr += tauId+','
        tauIdsStr = tauIdsStr[:-1] #remove last ","
        argv_toKeep = sys.argv
        sys.argv = [cfgName, tauIdsStr]
        handle = open(cfgName, 'r')
        cfo = imp.load_source('pycfg', cfgName, handle)
        cmsProcess = cfo.process
        handle.close()
        sys.argv = argv_toKeep
        out = open('PSetTmp.py','w')
        out.write(cmsProcess.dumpPython())
        out.close()
        config.JobType.psetName = 'PSetTmp.py'
        config.JobType.numCores = 4
    else:
        config.JobType.psetName = '../python/tauPi0Tree_cfg.py'
        config.JobType.numCores = 1
    config.JobType.disableAutomaticOutputCollection = True
    config.JobType.scriptExe = 'runAllSteps.py'
    params = None
    if prefetch:
        params = ['--prefetch']
    if tauIdsToRun:
        if params:
            params.append('--runtauids')
        else:
            params = ['--runtauids']
    config.JobType.scriptArgs = params
    config.JobType.inputFiles = inputFiles
    config.JobType.outputFiles = ['tauTreeForPi0Study.root']
    config.JobType.maxMemoryMB = 5000

    config.General.requestName = requestName
    config.General.workArea = "crab3_TauTrees"
    
    config.Data.inputDataset = inputDataset
    config.Data.inputDBS = "global" #FIXME, can be optional linked do dataset name (global or phys03)
    config.Data.outLFNDirBase = outLFNDirBase+outputDatasetTag
    config.Data.publication = True
    config.Data.outputDatasetTag = outputDatasetTag

    config.Site.storageSite = storage_element
    
    config.Data.splitting = 'EventAwareLumiBased'
    config.Data.unitsPerJob = eventsPerJob
    if numberOfJobs>0:
        config.Data.totalUnits = eventsPerJob*numberOfJobs
    else:
        config.Data.totalUnits = -1
    config.Data.ignoreLocality = ignoreLocality
    if whiteList:
        config.Site.whitelist = whiteList
    if blackList:
        config.Site.whitelist = blackList

    fullWorkDirName = config.General.workArea+"/crab_"+config.General.requestName
    requestExists = len(glob.glob(fullWorkDirName))!=0
    if requestExists:
        print("Request with name: {} exists. Skipping.".format(fullWorkDirName))
        return

    if runLocal:
        testDir = 'crab3_test/'
        os.makedirs(testDir, exist_ok=True)
        os.system('rm '+testDir+'/*')
        for f in inputFiles:
            os.system('cp -a '+f+' '+testDir)
        os.system('cp -a '+config.JobType.psetName+' '+testDir+'/PSet.py')
        os.system('cp -a '+config.JobType.scriptExe+' '+testDir)
        argStr = ''
        if config.JobType.scriptArgs:
            for a in config.JobType.scriptArgs:
                argStr += a+' '
        command = 'cd '+testDir+'; python3 ' +'runAllSteps.py '+argStr+' >& out_all_steps.log &'
        os.system(command)
    else:
        out = open('crabTmp.py','w')
        out.write(config.pythonise_())
        out.close()
        os.system("crab submit -c crabTmp.py")
        #os.system("crab submit -c crabTmp.py --dryrun")
        #crabCommand('submit', config = config)
    os.system("rm -f crabTmp.py* PSetTmp.py*")

#########################################
def addArguments(parser):
    parser.add_argument('-n', '--eventsPerJob', metavar='eventsPerJob', default=100000, type=int, help='Number of events per job that will be processed [Default: %(default)s]')
    parser.add_argument('-j', '--numberOfJobs', metavar='numberOfJobs', default=-1, type=int, help='Number of of job that will be run (-1: as many jobs to process all events) [Default: %(default)s]')
    parser.add_argument('-s', '--storageElement', metavar='storageElement', default='T2_PL_Swierk', help='Grid Storage Element [Default: %(default)s]')
    parser.add_argument('--outLFNDir', metavar='outLFNDir',  default='/store/user/bluj/PiZeroStudy/', help='LFN output directory (should be /store/user/your-user-name/[something]) [Default: %(default)s]')
    parser.add_argument('-i', '--inputDatasets', metavar='inputDataset', default=[], nargs='*', help='List of input datasets (if empty a hard-coded list will be used)')
    parser.add_argument('-m', '--mvaid', default=[], nargs='*', help='List of tauIds to be rerun, e.g. [deepTau2018v2p5, ...] [Default: %(default)s]')
    parser.add_argument('-v', '--version', default='v1', help='Set production version [Default: %(default)s]')
    parser.add_argument('--prefetch', default=False, action='store_true', help='Prefetch files for processing [Default: %(default)s]')
    parser.add_argument('--ignoreLocality', default=False, action='store_true', help='Ignore location of a dataset [Default: %(default)s]')
    parser.add_argument('--whiteList', nargs='*', help='White list of grid sites')
    parser.add_argument('--blackList', nargs='*', help='Black list of grid sites')
    parser.add_argument('--runLocal', default=False, action='store_true', help='Run locally (for tests only) [Default: %(default)s]')

#########################################
#########################################

if __name__ == '__main__':
    
    # command line arguments parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    addArguments(parser)
    args = parser.parse_args()

    inputDatasets = args.inputDatasets
    if not inputDatasets:        
        inputDatasets = [
            #"/GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v3/MINIAODSIM", #13M
            "/VBFHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM", #3M
            #"/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM", #100M
            #"/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1_ext1-v1/MINIAODSIM", #100M
            #"/TauGun_Pt-15to500_13p6TeV_pythia8/Run3Winter22MiniAOD-FlatPU0to70_122X_mcRun3_2021_realistic_v9-v3/MINIAODSIM", #1.7M
            #"/TauGun_Pt-15to500_13p6TeV_pythia8/Run3Winter22MiniAOD-FlatPU0to70_122X_mcRun3_2021_realistic_v9_ext1-v3/MINIAODSIM", #1.9M
        ]
        print('Predefined list of datasets will be used:',inputDatasets)
    eventsPerJob =  args.eventsPerJob
    numberOfJobs = args.numberOfJobs
    outLFNDirBase = args.outLFNDir
    storage_element = args.storageElement
    outputDatasetTag = args.version
    runLocal = args.runLocal
    prefetch = args.prefetch
    whiteList = args.whiteList
    blackList = args.blackList
    ignoreLocality = args.ignoreLocality
    tauIdsToRun = args.mvaid

########################################################
for inputDataset in inputDatasets:
    prepareCrabCfg(eventsPerJob = eventsPerJob,
                   numberOfJobs = numberOfJobs,
                   outLFNDirBase = outLFNDirBase,
                   storage_element = storage_element,
                   outputDatasetTag = outputDatasetTag,
                   inputDataset = inputDataset,
                   tauIdsToRun = tauIdsToRun,
                   prefetch = prefetch,
                   whiteList = whiteList,
                   blackList = blackList,
                   ignoreLocality = ignoreLocality,
                   runLocal = runLocal)

########################################################
