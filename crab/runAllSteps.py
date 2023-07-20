#!/usr/bin/env python

import sys
import os 

import argparse

#########################################
def addArguments(parser):
    parser.add_argument('-r', '--runtauids', default=False, action='store_true', help='Run tauIds on top of miniAOD [Default: %(default)s]')
    parser.add_argument('--prefetch', default=False, action='store_true', help='Prefetch files for processing [Default: %(default)s]')    

#########################################
if __name__ == '__main__':

    # command line arguments parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    addArguments(parser)
    args = parser.parse_args()

    runTauIds = args.runtauids
    prefetch = args.prefetch
    
    # First step: update PSet cfg provided by Crab in case of prefetch
    if prefetch:
        handle = open("PSet.py", 'r')
        exec(handle.read())
        cmsProcess = process
        handle.close()
        print("Prefetch input files to local directory", flush=True)
        fileNamesNew = []
        for f in cmsProcess.source.fileNames:
            if f.find("/store")>-1:
                fname = f[f.find("/store"):]
                print("\t copying "+fname, flush=True)
                fbase = os.path.basename(fname)
                command = "xrdcp -f -N root://xrootd-cms.infn.it//"+fname+" "+fbase
                os.system(command)
                fileNamesNew.append("file:"+fbase)
            else:
                print("file w/ name:",f,"not copied, skipping", flush=True)
        print("fileNamesLocal:",fileNamesNew, flush=True)
        cmsProcess.source.fileNames = fileNamesNew
        out = open("PSet_new.py","w")
        out.write(cmsProcess.dumpPython())
        out.close()
        command = "mv PSet_new.py PSet.py"
        os.system(command)

    # 2nd & 3rd steps: run tauIds if requested and produce trees
    if runTauIds:
        command = "cmsRun -j FrameworkJobReport.xml PSet.py"
        os.system(command)        
        command = "python3 produceTauTreeForPi0Study.py -i patTuple_newTauIDs.root -t slimmedTausNewID -m deepTau2017v2p1VSjet deepTau2017v2p1VSe deepTau2017v2p1VSmu deepTau2018v2p5VSjet deepTau2018v2p5VSe deepTau2018v2p5VSmu againstMuon3 MVADM"
        os.system(command)
    else:
        handle = open("PSet.py", 'r')
        exec(handle.read())
        cmsProcess = process
        handle.close()
        if not prefetch:
            fileNamesNew = []
            for f in cmsProcess.source.fileNames:
                if f.find("/store")>-1:
                    fname = 'root://xrootd-cms.infn.it/'+f[f.find("/store"):]
                    fileNamesNew.append(fname)
                else:
                    print("name of file:",f,"not updated, skipping", flush=True)
            print("fileNamesNew:",fileNamesNew, flush=True)
            cmsProcess.source.fileNames = fileNamesNew
        cmsProcess.tauPi0Tree.mvaid = [
            'deepTau2017v2p1VSjet', 'deepTau2017v2p1VSe', 'deepTau2017v2p1VSmu',
            'deepTau2018v2p5VSjet', 'deepTau2018v2p5VSe', 'deepTau2018v2p5VSmu',
            'againstMuon3', 'MVADM'
        ]
        out = open("PSet_new.py","w")
        out.write(cmsProcess.dumpPython())
        out.close()
        command = "mv PSet_new.py PSet.py"
        os.system(command)
        # run to produce framework report for crab
        command = "cmsRun -j FrameworkJobReport.xml PSet.py"
        os.system(command)
        # actual run
        command = "python3 produceTauTreeForPi0Study.py -c PSet.py"
        os.system(command)
