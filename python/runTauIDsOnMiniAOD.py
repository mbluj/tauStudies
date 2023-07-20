# Produce pat::Tau collection with the new DNN Tau-Ids from miniAOD 12Apr2018_94X_mc2017

import FWCore.ParameterSet.Config as cms

# Options
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.register('mvaIds',
                 'deepTau2018v2p5', #default
                 VarParsing.multiplicity.list, # singleton or list
                 VarParsing.varType.string, # string, int, or float
                 'List of tauIDs to produce')
options.parseArguments()

updatedTauName = "slimmedTausNewID"
minimalOutput = False
#eventsToProcess = -1
eventsToProcess = 1000
nThreads = 4

process = cms.Process('TauID')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')

# Input source
process.source = cms.Source('PoolSource', fileNames = cms.untracked.vstring(
    # File from dataset TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8
    #'/store/mc/RunIISummer20UL18MiniAOD/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v11_L1v1-v2/00000/009636D7-07B2-DB49-882D-C251FD62CCE7.root'
     #'file:/home/mbluj/work/data/MiniAOD/106X/GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18MiniAODv2_106X_upgrade2018_realistic_v16_L1v1-v3_15049D0D-61A2-A444-BFAA-509B741E58F8.root'
     '/store/mc/RunIISummer20UL18MiniAODv2/GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v3/100000/00DD4376-E5AA-A844-879B-F8CD8FB7195D.root'
))

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(eventsToProcess) )

# Add new TauIDs
import RecoTauTag.RecoTau.tools.runTauIdMVA as tauIdConfig
#toKeep = [ #"2017v2", "dR0p32017v2", "newDM2017v2",
#           # "deepTau2017v1",
#           #"deepTau2017v2p1",
#           "deepTau2018v2p5",
#           # "DPFTau_2016_v0",
#           # "DPFTau_2016_v1",
#           #"againstEle2018",
#           'MVADM_2017_v1'
#           ]
toKeep = options.mvaIds
print('tauIds to produce:', toKeep)
tauIdEmbedder = tauIdConfig.TauIDEmbedder(process, debug = False,
                    updatedTauName = updatedTauName,
                    toKeep = toKeep)
tauIdEmbedder.runTauID()

# Output definition
process.out = cms.OutputModule("PoolOutputModule",
     fileName = cms.untracked.string('patTuple_newTauIDs.root'),
     compressionAlgorithm = cms.untracked.string('LZMA'),
     compressionLevel = cms.untracked.int32(4),
     outputCommands = cms.untracked.vstring('drop *')
)
if not minimalOutput:
     print("Store full MiniAOD EventContent")
     from Configuration.EventContent.EventContent_cff import MINIAODSIMEventContent
     from PhysicsTools.PatAlgos.slimming.MicroEventContent_cff import MiniAODOverrideBranchesSplitLevel
     process.out.outputCommands = MINIAODSIMEventContent.outputCommands
     process.out.overrideBranchesSplitLevel = MiniAODOverrideBranchesSplitLevel
process.out.outputCommands.append("keep *_"+updatedTauName+"_*_*")

# Path and EndPath definitions
process.p = cms.Path(
    process.rerunMvaIsolationSequence *
    getattr(process,updatedTauName)
)
process.endjob = cms.EndPath(process.endOfProcess)
process.outpath = cms.EndPath(process.out)
# Schedule definition
process.schedule = cms.Schedule(process.p,process.endjob,process.outpath)

##
process.load('FWCore.MessageLogger.MessageLogger_cfi')
if process.maxEvents.input.value()>10:
     process.MessageLogger.cerr.FwkReport.reportEvery = process.maxEvents.input.value()//10
if process.maxEvents.input.value()>10000 or process.maxEvents.input.value()<0:
     process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.options = cms.untracked.PSet(
     wantSummary = cms.untracked.bool(False),
     numberOfThreads = cms.untracked.uint32(nThreads),
     numberOfStreams = cms.untracked.uint32(0)
)
