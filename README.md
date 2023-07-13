# tauStudies

## Installation
```
cmsrel CMSSW_12_4_5 #should work also with newer releases
cd CMSSW_12_4_5/src
cmsenv
git cms-init #it is optional (useful to install CMSSW updates)
git clone https://github.com/mbluj/tauStudies.git
```
---
## Ntuple production
```
cd tauStudies/python/
python3 produceTauTreeForPi0Study.py -i inputFiles #use --help to consult available options, e.g.
python3 produceTauTreeForPi0Study.py -i /eos/cms/store/mc/RunIISummer20UL18MiniAODv2/GluGluHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v3/100000/*.root -t slimmedTaus -n -1 -o tauTreeForPi0Study_ggH125_Summer20UL18.root --mvaid MVADM againstMuon3 deepTau2018v2p5VSe deepTau2018v2p5VSmu deepTau2018v2p5VSjet deepTau2017v2p1VSe deepTau2017v2p1VSmu deepTau2017v2p1VSjet
```

## Data preparation for ML tools
```
cd tauStudies/python/ML
python3 DataPreparation.py # edit it to define input files used
```

## XGBoost model training
```
cd tauStudies/python/ML
python3 Training_XGB_phishift.py # edit it to define input files used
```

## XGBoost model testing
```
cd tauStudies/python/ML
python3 Test_XGB_phishift.py # edit it to define input files used
```

---
## TODOs
* Provide crab tools
* Add posibility to configure ntuplizer with a PSet (useful to run with crab)
* Use command-line argument to define input files for ML tools
* Improve ML model
* ...
