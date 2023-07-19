import FWCore.ParameterSet.Config as cms

process = cms.Process("TauTreeProd")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# needed only to define list of input files (and for crab)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:miniAOD.root'
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
