'''
Simplified version of relValTools.py from TauReleaseValidation(unnecessary tools removed)
''' 

import re
import os
import subprocess

globaldebug = False


def addArguments(parser):
    parser.add_argument('-i','--inputfiles', required=True, nargs='*', help="List of files (required)")
    parser.add_argument('-n', '--maxEvents', default=-1, type=int, help='Number of events that will be analyzed (-1 = all events) [Default: %(default)s]')
    parser.add_argument('--debug', default=False, help="Debug option [Default: %(default)s]", action="store_true")
    parser.add_argument('-o', '--outputFileName', default='tauTreeForPi0Study.root', help="Output file name [Default: %(default)s]")
    parser.add_argument('-m', '--mvaid', default=[], nargs='*',
                        help="Select mvaIDs that should be obtained via rerunning TAUId sequence, e.g. [2017v1, 2017v2, newDM2017v2, dR0p32017v2, 2016v1, newDM2016v1]. [Default: %(default)s]")
    parser.add_argument('-t', '--tauCollection', default='slimmedTaus', help="Tau collection to be used. [Default: %(default)s].")
    parser.add_argument('--addAntiLepton', default=False, action='store_true', help='Access classic anti-lepton discriminators (can be not present in recent samples)')
    parser.add_argument('--addMVAIso', default=False, action='store_true', help='Access MVAIso discriminators (can be not present in recent samples)')


def dprint(*text):
    if globaldebug and text is not None:
        for t in text:
            print (t,)
        print()
        # print " ".join(map(str, text))


def dpprint(*text):
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    if globaldebug and text is not None:
        for t in text:
            pp.pprint(t)
        # pp.pprint(" \n".join(map(str, text)))


def get_cmssw_version():
    """returns 'CMSSW_X_Y_Z'"""
    return os.environ["CMSSW_RELEASE_BASE"].split('/')[-1]


def get_cmssw_version_number():
    """returns 'X_Y_Z' (without 'CMSSW_')"""
    return map(int, get_cmssw_version().split("CMSSW_")[1].split("_")[0:3])


def versionToInt(release=9, subversion=4, patch=0):
    return release * 10000 + subversion * 100 + patch


def is_above_cmssw_version(release=9, subversion=4, patch=0):
    split_cmssw_version = get_cmssw_version_number()
    if versionToInt(release, subversion, patch) > versionToInt(split_cmssw_version[0], split_cmssw_version[1], split_cmssw_version[2]):
        return False
    return True
