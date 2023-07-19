import FWCore.ParameterSet.Config as cms

process = cms.Process("TauTreeProd")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# needed only to define list of input files (and for crab)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:miniAOD.root'
        #'/store/mc/RunIISummer20UL18MiniAODv2/GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v3/100000/00DD4376-E5AA-A844-879B-F8CD8FB7195D.root'
    ),
    #skipEvents = cms.untracked.uint32(5)
)

process.tauPi0Tree = cms.PSet(
    mvaid = cms.vstring(),
    tauCollection = cms.InputTag('slimmedTaus'),
    addAntiLepton = cms.bool(False),
    addMVAIso = cms.bool(False),
    debug = cms.bool(False)
)

# needed only to define name of output file (and for crab)
process.TFileService = cms.Service("TFileService",
  fileName = cms.string('tauTreeForPi0Study.root')
)

##
process.load('FWCore.MessageLogger.MessageLogger_cfi')
if process.maxEvents.input.value()>10:
     process.MessageLogger.cerr.FwkReport.reportEvery = process.maxEvents.input.value()//10
if process.maxEvents.input.value()>10000 or process.maxEvents.input.value()<0:
     process.MessageLogger.cerr.FwkReport.reportEvery = 1000
