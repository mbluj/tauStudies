#!/usr/bin/env python3

''' Produces a flat tree for tau pi0 studies; based on produceTauValTree.py from TauReleaseValidation
Authors: Michal Bluj
'''

import math
import sys
import os
import copy
import subprocess
from time import time
from datetime import datetime, timedelta

import ROOT
import argparse  # it needs to come after ROOT import

from DataFormats.FWLite import Events, Handle
from PhysicsTools.HeppyCore.utils.deltar import deltaR, bestMatch, deltaR2, \
    deltaPhi
from PhysicsTools.Heppy.physicsutils.TauDecayModes import tauDecayModes

from Var import Var
from tau_ids import basic_tau_ids, mvaiso_tau_ids, lepton_tau_ids, \
    tau_ids, fill_tau_ids


from relValTools import addArguments, dprint

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)

########################
def finalDaughters(gen, daughters=None):
    if daughters is None:
        daughters = []
    for i in range(gen.numberOfDaughters()):
        daughter = gen.daughter(i)
        if daughter.numberOfDaughters() == 0:
            daughters.append(daughter)
        else:
            finalDaughters(daughter, daughters)

    return daughters

########################
def visibleP4(gen):
    gen.final_ds = finalDaughters(gen)
    return sum(
        (d.p4() for d in gen.final_ds
            if abs(d.pdgId()) not in [12, 14, 16]),
        ROOT.math.XYZTLorentzVectorD()
    )

########################
def removeOverlap(all_jets, gen_leptons, dR2=0.25): # dR2=0.25  ==  dR=0.5
    non_tau_jets = []
    for j_cand in all_jets:
        if not any(deltaR2(j_cand, lep) < dR2 for lep in gen_leptons):
            non_tau_jets.append(j_cand)

    return non_tau_jets

########################
def isGenLepton(lep_cand, pid, min_pt=20, max_eta=2.3):
    # more relaxed definition of leptons faking taus:
    # select also particles that radiated
    # and would otherwise fail isPromptFinalState()
    return (
        abs(lep_cand.pdgId()) == pid and
        (
            lep_cand.statusFlags().isPrompt() or
            lep_cand.isDirectPromptTauDecayProductFinalState()
        ) and
        lep_cand.pt() > min_pt and
        abs(lep_cand.eta()) < max_eta
    )

########################
def MatchTausToJets(refObjs, dr_max=0.5):

  _dr2_max = dr_max*dr_max
  # For each Jet, get the closest RecoTau
  Match = {}
  for jetidx,refObj in enumerate(refObjs):
      tau, _dr2_ = bestMatch(refObj, taus)
      for tauidx,itau in enumerate(taus):
        if itau==tau: break
      if _dr2_ < _dr2_max: Match[jetidx]=tauidx

  # Is the same Tau assinged to more than one Jet?
  DoubleCheck = []
  for ijet,itau in iter(Match.items()):
    for jjet,jtau in iter(Match.items()):
      if jjet >= ijet: continue
      if itau==jtau:
        if ijet not in DoubleCheck: DoubleCheck.append(ijet)
        if jjet not in DoubleCheck: DoubleCheck.append(jjet)

  # Get all distances between all conflicting Jets and corresponding Taus
  Distances = {}
  for ijet in DoubleCheck:
    for jjet in DoubleCheck:
      itau = Match[jjet]
      Distances[str(ijet)+"_"+str(itau)] = deltaR(taus[itau].eta(), taus[itau].phi(), refObjs[ijet].eta(), refObjs[ijet].phi())
  #print Distances

  # Remove all conflicting Jets, to re-assign later
  for ijet in DoubleCheck:
    del Match[ijet]

  # Assign shortest distance between Tau and Jet, then move on ignoring the already assigned Taus/Jets
  while Distances != {}:
    keepthis = min(Distances, key=Distances.get)
    thisjet = int(keepthis[:keepthis.find("_")])
    thistau = int(keepthis[keepthis.rfind("_")+1:])
    Match[thisjet] = thistau
    deletethis = []
    for element in Distances:
      if element.startswith(str(thisjet)) or element.endswith(str(thistau)): deletethis.append(element)
    for element in deletethis: del Distances[element]

  return Match

########################
def MatchGenToTaus(genObjs, dr_max=0.5):

  _dr2_max = dr_max*dr_max
  # For each recoTau, get the closest genObj
  Match = {}
  for tauidx,tau in enumerate(taus):
      genObj, _dr2_ = bestMatch(tau, genObjs)
      for genidx,igen in enumerate(genObjs):
        if igen==genObj: break
      if _dr2_ < _dr2_max: Match[tauidx]=genidx
      # assume not double assigment (can apply cleaning as above, but for dR<0.5 it should be reasonable assumption

  return Match

########################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    addArguments(parser)
    args = parser.parse_args()

    globaldebug = args.debug
    maxEvents = args.maxEvents
    skipEvents = args.skipEvents
    tauCollection = args.tauCollection
    mvaid = args.mvaid
    add_anti_lepton = args.addAntiLepton
    add_mva_iso = args.addMVAIso
    inputfiles = args.inputfiles
    outputFileName = args.outputFileName

    cfg = args.cfg
    process = None
    if cfg != None:
        print("Configuration from cfg file:", cfg)
        f_cfg = open(cfg)
        exec(f_cfg.read())
        f_cfg.close()
    if process != None:
        if hasattr(process,'maxEvents') and hasattr(process.maxEvents,'input'):
            maxEvents = process.maxEvents.input.value()
        if hasattr(process,'source'):
            psource = process.source
            if hasattr(psource,'fileNames'):
                inputfiles = psource.fileNames.value()
            if hasattr(psource,'skipEvents'):
                skipEvents = psource.skipEvents.value()
        if hasattr(process,'TFileService') and hasattr(process.TFileService,'fileName'):
            outputFileName = process.TFileService.fileName.value()
        if hasattr(process,'tauPi0Tree'):
            treeConf = process.tauPi0Tree
            if hasattr(treeConf,'mvaid'):
                mvaid = treeConf.mvaid.value()
            if hasattr(treeConf,'tauCollection'):
                tauCollection = treeConf.tauCollection.value()
            if hasattr(treeConf,'addAntiLepton'):
                add_anti_lepton = treeConf.addAntiLepton.value()
            if hasattr(treeConf,'addMVAIso'):
                add_mva_iso = treeConf.addMVAIso.value()
            if hasattr(treeConf,'debug'):
                globaldebug = treeConf.debug.value()

    filelist = []
    if inputfiles:
        filelist = inputfiles
    else:
        print('Please provide nonempty list of input files')
        sys.exit(0)

    events = Events(filelist)
    if maxEvents < 0: maxEvents=-1
    print(len(filelist), "files will be analyzed:", filelist, '\nEvents will be analyzed: %i' % maxEvents)

    # +++++++ Output file +++++++++
    if not outputFileName:
        outputFileName = 'tauTreeForPi0Study.root'
        print('Name of output file is not specified, ' \
              'use default name:', outputFileName)
    if outputFileName[-5:] != ".root":
        outputFileName += '.root'
        print("output file should have a root format" \
              " - added automatically:", outputFileName)

    print("outputFileName:", outputFileName)

    out_file = ROOT.TFile(outputFileName, 'recreate')

    # +++++++ Useful constants, thresholds, etc. +++++++++
    puppiMin = 0.#1 #minimal puppi weight
    pi0RecoM = 0.136 #approximate pi0 mass from early fits in PF paper
    pi0RecoW = 0.013 #approximate width of pi0 peak from early fits in PF paper
    pi0M = 0.135

    # +++++++ Histrams and tree +++++++++
    h_ntau = ROOT.TH1F("h_ntau", "h_ntau", 20, 0, 20)

    tau_tree = ROOT.TTree('per_tau', 'per_tau')

    all_vars = [        
        Var('tau_eventid', int),
        Var('tau_id', int),
        Var('tau_run', int),
        Var('tau_lumi', int),

        Var('tau_vertex', int), #no. of vertices
        Var('tau_rho', float),
        Var('tau_nTruePU', float),
        Var('tau_nPU', int),
        
        Var('tau_match', int),
        Var('tau_dm', int),
        Var('tau_pt', float),
        Var('tau_eta', float),
        Var('tau_phi', float),
        Var('tau_mass', float),
        Var('tau_chargedpt', float), #MB split on prongs?
        Var('tau_chargedeta', float), #MB split on prongs?
        Var('tau_chargedphi', float), #MB split on prongs?
        Var('tau_leadTrackpt', float), #MB pt of leading track (not cand)
        Var('tau_leadCaloEt', float), #MB pt of leading track (not cand)
        Var('tau_leadEcalEt', float), #MB pt of leading track (not cand)
        Var('tau_leadPdgId', int), #MB lead or for all charged?

        Var('tau_neutralpt', float),
        Var('tau_neutraleta', float),
        Var('tau_neutralphi', float),
        # different pi0 hypotheses
        #leading photon (not need to store as all photons (up to 4) stored independently)
        #two leading photons (or just 1st one in case of nGamma==1)
        Var('tau_pi0pt2', float),
        Var('tau_pi0eta2', float),
        Var('tau_pi0phi2', float),
        Var('tau_pi0mass2', float),
        #all photons (up to 4)
        Var('tau_pi0pt3', float),
        Var('tau_pi0eta3', float),
        Var('tau_pi0phi3', float),
        Var('tau_pi0mass3', float),
        # selected from leading photons "myGuess"
        Var('tau_pi0pt4', float),
        Var('tau_pi0eta4', float),
        Var('tau_pi0phi4', float),
        Var('tau_pi0mass4', float),
        Var('tau_nGammas', int),
        Var('tau_nGammasUsed', int),
        Var('tau_nEle', int), #MB global nEle or isEle for each pf-photon (stip members)
        Var('tau_gendm', int),
        Var('tau_genpt', float),
        Var('tau_geneta', float),
        Var('tau_genphi', float),
        Var('tau_genmass', float),
        Var('tau_genchargedpt', float), #MB split on prongs?
        Var('tau_genchargedeta', float), #MB split on prongs?
        Var('tau_genchargedphi', float), #MB split on prongs?
        Var('tau_genchargedmass', float), #MB split on prongs?
        Var('tau_genneutralpt', float), #MB add all gammas/pi0's?
        Var('tau_genneutraleta', float), #MB add all gammas/pi0's?
        Var('tau_genneutralphi', float), #MB add all gammas/pi0's?
        Var('tau_genneutralmass', float), #MB add all gammas/pi0's?
        Var('tau_genpi0pt_1', float),
        Var('tau_genpi0eta_1', float),
        Var('tau_genpi0phi_1', float),
        Var('tau_genpi0mass_1', float),
        Var('tau_genpi0pt_2', float),
        Var('tau_genpi0eta_2', float),
        Var('tau_genpi0phi_2', float),
        Var('tau_genpi0mass_2', float),
        Var('tau_gengammapt_1', float),
        Var('tau_gengammaeta_1', float),
        Var('tau_gengammaphi_1', float),
        Var('tau_gengammapt_2', float),
        Var('tau_gengammaeta_2', float),
        Var('tau_gengammaphi_2', float),
        Var('tau_gengammapt_3', float),
        Var('tau_gengammaeta_3', float),
        Var('tau_gengammaphi_3', float),
        Var('tau_gengammapt_4', float),
        Var('tau_gengammaeta_4', float),
        Var('tau_gengammaphi_4', float),

        # Var('tau_vtxTovtx_dz', float),
        Var('tau_tauVtxTovtx_dz', float), #MB needed?
        Var('tau_tauVtxIdx', int),
        Var('tau_dzPV', float),

        Var('tau_dxy', float),
        Var('tau_dxy_err', float),
        Var('tau_dxy_sig', float),
        Var('tau_ip3d', float),
        Var('tau_ip3d_err', float),
        Var('tau_ip3d_sig', float),
        Var('tau_flightLength', float),
        Var('tau_flightLength_sig', float),
    ]

    #individual gammas and their combinations (up to 4)
    for iGam in range(0,4):
        all_vars.append(Var('tau_gammapt'+str(iGam+1), float))
        all_vars.append(Var('tau_gammaeta'+str(iGam+1), float))
        all_vars.append(Var('tau_gammaphi'+str(iGam+1), float))
        all_vars.append(Var('tau_gammadeta'+str(iGam+1), float))
        all_vars.append(Var('tau_gammadphi'+str(iGam+1), float))
        for jGam in range(iGam+1,4):
            all_vars.append(Var('tau_gammapt'+str(iGam+1)+str(jGam+1), float))
            all_vars.append(Var('tau_gammaeta'+str(iGam+1)+str(jGam+1), float))
            all_vars.append(Var('tau_gammaphi'+str(iGam+1)+str(jGam+1), float))
            all_vars.append(Var('tau_gammadeta'+str(iGam+1)+str(jGam+1), float))
            all_vars.append(Var('tau_gammadphi'+str(iGam+1)+str(jGam+1), float))
            all_vars.append(Var('tau_gammamass'+str(iGam+1)+str(jGam+1), float))
            for kGam in range(jGam+1,4):
                all_vars.append(Var('tau_gammapt'+str(iGam+1)+str(jGam+1)+str(kGam+1), float))
                all_vars.append(Var('tau_gammaeta'+str(iGam+1)+str(jGam+1)+str(kGam+1), float))
                all_vars.append(Var('tau_gammaphi'+str(iGam+1)+str(jGam+1)+str(kGam+1), float))
                all_vars.append(Var('tau_gammadeta'+str(iGam+1)+str(jGam+1)+str(kGam+1), float))
                all_vars.append(Var('tau_gammadphi'+str(iGam+1)+str(jGam+1)+str(kGam+1), float))
                all_vars.append(Var('tau_gammamass'+str(iGam+1)+str(jGam+1)+str(kGam+1), float))

    # tauIDs
    all_tau_ids = copy.deepcopy(basic_tau_ids)
    
    if add_anti_lepton:
        all_tau_ids += lepton_tau_ids

    if add_mva_iso:
        all_tau_ids += mvaiso_tau_ids

    for mva_id in mvaid:
        all_tau_ids += tau_ids[mva_id]

    for (tau_id, v_type) in all_tau_ids:
        all_vars.append(Var('tau_' + tau_id, v_type))

    all_var_dict = {var.name: var for var in all_vars}

    for var in all_vars:
        tau_tree.Branch(var.name, var.storage, var.name +
                        '/' + ('I' if var.type == int else 'D'))

    evtid = 0

    NMatchedTaus = 0

    tauH = Handle('vector<pat::Tau>')
    vertexH = Handle('std::vector<reco::Vertex>')
    genParticlesH = Handle('std::vector<reco::GenParticle>')
    jetH = Handle('vector<pat::Jet>')
    genJetH = Handle('vector<reco::GenJet>')
    puH = Handle('std::vector<PileupSummaryInfo>')
    candH = Handle('vector<pat::PackedCandidate>')
    lostH = Handle('vector<pat::PackedCandidate>')
    lostEleH = Handle('vector<pat::PackedCandidate>')
    rhoH = Handle('double')

    start = time()
    for event in events:
        if skipEvents>0:
            skipEvents -= 1
            continue;
        evtid += 1
        eid = event.eventAuxiliary().id().event()
        run = event.eventAuxiliary().run()
        lumi = event.eventAuxiliary().luminosityBlock()

        if evtid % 1000 == 0:
            if maxEvents>0:
                percentage = float(evtid)/maxEvents*100.
                speed = float(evtid)/(time()-start)
                ETA = datetime.now() + timedelta(seconds=(maxEvents-evtid) / max(0.1, speed))
                print ('===> processing %d / %d event \t completed %.1f%s \t %.1f ev/s \t ETA %s s' %(evtid, maxEvents, percentage, '%', speed, ETA.strftime('%Y-%m-%d %H:%M:%S')))
            else:
                print ('===> processing %d event' %(evtid))
        if maxEvents > 0 and evtid > maxEvents:
            evtid -= 1 #for correct reporting
            break

        event.getByLabel(tauCollection, tauH)
        event.getByLabel("offlineSlimmedPrimaryVertices", vertexH)
        event.getByLabel("slimmedAddPileupInfo", puH)
        event.getByLabel('prunedGenParticles', genParticlesH)
        event.getByLabel('slimmedGenJets', genJetH)
        event.getByLabel('fixedGridRhoFastjetAll', rhoH)

        event.getByLabel("packedPFCandidates", candH)
        pfCands = candH.product()
        event.getByLabel("lostTracks", lostH)
        lostCands = lostH.product()
        event.getByLabel("lostTracks:eleTracks", lostEleH)
        eleTracks = lostEleH.product()
        #print("#lost:",len(lostCands),"#eleTracks",len(eleTracks))

        taus = tauH.product()
        vertices = vertexH.product()
        puInfo = puH.product()
        genParticles = genParticlesH.product()
        genJets = genJetH.product()
        rho = rhoH.product()[0] #trick to convert adress to number

        genTaus = [p for p in genParticles if abs(p.pdgId()) == 15 and 
                   p.isPromptDecayed() and p.isLastCopy()]
        #add visible p4, visible decay products and DM
        genTausSelected = []
        for gen_tau in genTaus:
            gen_tau.visP4 = visibleP4(gen_tau)
            gen_tau.dm = tauDecayModes.genDecayModeInt(
                [d for d in gen_tau.final_ds
                 if abs(d.pdgId()) not in [12, 14, 16]]
            )
            pt_min = 15
            if gen_tau.dm == -11 or gen_tau.dm == -13: pt_min = 8
            if gen_tau.visP4.pt() > pt_min and abs(gen_tau.visP4.eta()) < 3:
                genTausSelected.append(gen_tau)

        genElectrons = [
            p for p in genParticles if isGenLepton(p, 11, 8, 3) and p.statusFlags().isPrompt()]
        genMuons = [
            p for p in genParticles if isGenLepton(p, 13, 8, 3) and p.statusFlags().isPrompt()]

        genLeptons = genTausSelected + genElectrons + genMuons

        genJets = [
            j for j in genJets if j.pt() > 15 and abs(j.eta()) < 3] 

        genJetsCleaned = removeOverlap(genJets, genLeptons, dR2=0.2*0.2)
        
        allGenObjs = genTausSelected + genElectrons + genMuons + genJetsCleaned

        Matched = MatchGenToTaus(allGenObjs, 0.2)

        ###
        h_ntau.Fill(len(taus))
        for tauidx,tau in enumerate(taus):
            # reset vars
            for var in all_vars:
                var.reset()
            # event info
            all_var_dict['tau_id'].fill(evtid)
            all_var_dict['tau_eventid'].fill(eid)
            all_var_dict['tau_run'].fill(run)
            all_var_dict['tau_lumi'].fill(lumi)
            all_var_dict['tau_vertex'].fill(len(vertices))
            all_var_dict['tau_rho'].fill(rho)
            for iPuInfo in puInfo:
                if iPuInfo.getBunchCrossing() == 0:
                    all_var_dict['tau_nTruePU'].fill(
                        iPuInfo.getTrueNumInteractions())
                    all_var_dict['tau_nPU'].fill(
                        iPuInfo.getPU_NumInteractions())
                    break
            # match info
            if tauidx in Matched:
                gen = allGenObjs[Matched[tauidx]]
                if abs(gen.pdgId())==15:
                    all_var_dict['tau_gendm'].fill(gen.dm)
                    all_var_dict['tau_genpt'].fill(gen.visP4.pt())
                    all_var_dict['tau_geneta'].fill(gen.visP4.eta())
                    all_var_dict['tau_genphi'].fill(gen.visP4.phi())
                    all_var_dict['tau_genmass'].fill(gen.visP4.mass())
                    charged_p4 = sum(
                        (d.p4() for d in gen.final_ds
                         if d.charge()),
                        ROOT.math.XYZTLorentzVectorD())
                    neutral_p4 = sum(
                        (d.p4() for d in gen.final_ds
                         if (abs(d.pdgId()) not in [12, 14, 16] and
                             not d.charge())),
                        ROOT.math.XYZTLorentzVectorD())
                    all_var_dict['tau_genchargedpt'].fill(charged_p4.pt())
                    all_var_dict['tau_genchargedeta'].fill(charged_p4.eta())
                    all_var_dict['tau_genchargedphi'].fill(charged_p4.phi())
                    all_var_dict['tau_genchargedmass'].fill(charged_p4.mass())
                    all_var_dict['tau_genneutralpt'].fill(neutral_p4.pt())
                    all_var_dict['tau_genneutraleta'].fill(neutral_p4.eta())
                    all_var_dict['tau_genneutralphi'].fill(neutral_p4.phi())
                    all_var_dict['tau_genneutralmass'].fill(neutral_p4.mass())
                    #taus with one pi0
                    if(gen.dm>=0 and gen.dm%5==1):
                        gamma_ds = [d for d in gen.final_ds if abs(d.pdgId()) == 22]
                        if len(gamma_ds)!=2:
                            print("Warning: gendm =",gen.dm,"#gammas =",len(gamma_ds)," != 2!!!")
                        else:
                            pi0_1 = gamma_ds[0].p4()+gamma_ds[1].p4()
                            all_var_dict['tau_genpi0pt_1'].fill(pi0_1.pt())
                            all_var_dict['tau_genpi0eta_1'].fill(pi0_1.eta())
                            all_var_dict['tau_genpi0phi_1'].fill(pi0_1.phi())
                            all_var_dict['tau_genpi0mass_1'].fill(pi0_1.mass())
                            all_var_dict['tau_gengammapt_1'].fill(gamma_ds[0].pt())
                            all_var_dict['tau_gengammaeta_1'].fill(gamma_ds[0].eta())
                            all_var_dict['tau_gengammaphi_1'].fill(gamma_ds[0].phi())
                            all_var_dict['tau_gengammapt_2'].fill(gamma_ds[1].pt())
                            all_var_dict['tau_gengammaeta_2'].fill(gamma_ds[1].eta())
                            all_var_dict['tau_gengammaphi_2'].fill(gamma_ds[1].phi())
                    #taus with two pi0s
                    elif(gen.dm>=0 and gen.dm%5==2):
                        gamma_ds = [d for d in gen.final_ds if abs(d.pdgId()) == 22]
                        if len(gamma_ds)!=4:
                            print("Warning: gendm =",gen.dm,"#gammas =",len(gamma_ds)," != 4!!!")
                        else:
                            pi0_1 = gamma_ds[0].p4()+gamma_ds[1].p4()
                            pi0_2 = gamma_ds[2].p4()+gamma_ds[3].p4()
                            i1=0; i2=1; i3=2; i4=3
                            m1234 = abs(pi0_1.mass()-pi0M)+abs(pi0_2.mass()-pi0M)
                            m1324 = abs((gamma_ds[0].p4()+gamma_ds[2].p4()).mass()-pi0M)+abs((gamma_ds[1].p4()+gamma_ds[3].p4()).mass()-pi0M)
                            m1423 = abs((gamma_ds[0].p4()+gamma_ds[3].p4()).mass()-pi0M)+abs((gamma_ds[1].p4()+gamma_ds[2].p4()).mass()-pi0M)
                            if(m1324<m1234 and m1324<m1423):
                                pi0_1 = gamma_ds[0].p4()+gamma_ds[2].p4()
                                pi0_2 = gamma_ds[1].p4()+gamma_ds[3].p4()
                                i1=0; i2=2; i3=1; i4=3
                            elif(m1423<m1234 and m1423<m1324):
                                pi0_1 = gamma_ds[0].p4()+gamma_ds[3].p4()
                                pi0_2 = gamma_ds[1].p4()+gamma_ds[2].p4()
                                i1=0; i2=3; i3=1; i4=2
                            all_var_dict['tau_genpi0pt_1'].fill(pi0_1.pt())
                            all_var_dict['tau_genpi0eta_1'].fill(pi0_1.eta())
                            all_var_dict['tau_genpi0phi_1'].fill(pi0_1.phi())
                            all_var_dict['tau_genpi0mass_1'].fill(pi0_1.mass())
                            all_var_dict['tau_genpi0pt_2'].fill(pi0_2.pt())
                            all_var_dict['tau_genpi0eta_2'].fill(pi0_2.eta())
                            all_var_dict['tau_genpi0phi_2'].fill(pi0_2.phi())
                            all_var_dict['tau_genpi0mass_2'].fill(pi0_2.mass())
                            all_var_dict['tau_gengammapt_1'].fill(gamma_ds[i1].pt())
                            all_var_dict['tau_gengammaeta_1'].fill(gamma_ds[i1].eta())
                            all_var_dict['tau_gengammaphi_1'].fill(gamma_ds[i1].phi())
                            all_var_dict['tau_gengammapt_2'].fill(gamma_ds[i2].pt())
                            all_var_dict['tau_gengammaeta_2'].fill(gamma_ds[i2].eta())
                            all_var_dict['tau_gengammaphi_2'].fill(gamma_ds[i2].phi())
                            all_var_dict['tau_gengammapt_3'].fill(gamma_ds[i3].pt())
                            all_var_dict['tau_gengammaeta_3'].fill(gamma_ds[i3].eta())
                            all_var_dict['tau_gengammaphi_3'].fill(gamma_ds[i3].phi())
                            all_var_dict['tau_gengammapt_4'].fill(gamma_ds[i4].pt())
                            all_var_dict['tau_gengammaeta_4'].fill(gamma_ds[i4].eta())
                            all_var_dict['tau_gengammaphi_4'].fill(gamma_ds[i4].phi())


                    if gen.dm == -11:
                        all_var_dict['tau_match'].fill(3)
                    elif gen.dm == -13:
                        all_var_dict['tau_match'].fill(4)
                    else:
                        all_var_dict['tau_match'].fill(5)
                        NMatchedTaus += 1
                else:
                    all_var_dict['tau_genpt'].fill(gen.pt())
                    all_var_dict['tau_geneta'].fill(gen.eta())
                    all_var_dict['tau_genphi'].fill(gen.phi())
                    all_var_dict['tau_genmass'].fill(gen.mass())
                    if abs(gen.pdgId())==11:
                        all_var_dict['tau_match'].fill(1)
                    elif abs(gen.pdgId())==13:
                        all_var_dict['tau_match'].fill(2)
                    else:
                        all_var_dict['tau_match'].fill(6)

            all_var_dict['tau_dm'].fill(tau.decayMode())
            all_var_dict['tau_pt'].fill(tau.pt())
            all_var_dict['tau_eta'].fill(tau.eta())
            all_var_dict['tau_phi'].fill(tau.phi())
            all_var_dict['tau_mass'].fill(tau.mass())

            tau_charged_p4 = sum((d.p4() for d in tau.signalChargedHadrCands()),
                                 ROOT.math.XYZTLorentzVectorD())
            all_var_dict['tau_chargedpt'].fill(tau_charged_p4.pt())
            all_var_dict['tau_chargedeta'].fill(tau_charged_p4.eta())
            all_var_dict['tau_chargedphi'].fill(tau_charged_p4.phi())
            tau_neutral_p4 = sum((d.p4() for d in tau.signalGammaCands()),
                                 ROOT.math.XYZTLorentzVectorD())
            all_var_dict['tau_neutralpt'].fill(tau_neutral_p4.pt())
            all_var_dict['tau_neutraleta'].fill(tau_neutral_p4.eta())
            all_var_dict['tau_neutralphi'].fill(tau_neutral_p4.phi())

            if len(vertices)>0:
                all_var_dict['tau_dzPV'].fill(tau.leadChargedHadrCand().dz(vertices[0].position()))
            leadChHadr = tau.leadChargedHadrCand()
            all_var_dict['tau_leadPdgId'].fill(leadChHadr.pdgId())
            all_var_dict['tau_leadCaloEt'].fill(leadChHadr.caloFraction()*leadChHadr.pt())
            all_var_dict['tau_leadEcalEt'].fill(leadChHadr.caloFraction()*leadChHadr.pt()*(1.-leadChHadr.hcalFraction()))
            if abs(leadChHadr.pdgId())!=11:
                all_var_dict['tau_leadTrackpt'].fill(leadChHadr.ptTrk())
            else:
                isKF = False
                if globaldebug: print("Looking for KFTrack of leading pf-electron")
                for eleTrack in eleTracks:
                    deta = abs(leadChHadr.etaAtVtx() - eleTrack.etaAtVtx())
                    dphi = abs(deltaPhi(leadChHadr.phiAtVtx(), eleTrack.phiAtVtx()))
                    if deta<5e-3 and dphi<5e-2:
                        if globaldebug: print("\tKFtrack found, deta, dphi:",deta,dphi)
                        all_var_dict['tau_leadTrackpt'].fill(eleTrack.ptTrk())
                        isKF = True
                        break
                if not isKF:
                    if globaldebug: print("\tKFtrack not found")
                    all_var_dict['tau_leadTrackpt'].fill(leadChHadr.ptTrk())

            # Use candidate to vertex associaton as in MiniAOD
            # (it can be different than vertex closest in dz to lead track)
            tau_vertex_idxpf = tau.leadChargedHadrCand().vertexRef().key()
            # or uncomment code below to look for vertex closest in dz
            '''
            tau_dzToVtx = 99
            for i, vertex in enumerate(vertices):
                dz = abs(tau.leadChargedHadrCand().dz(vertex.position()))
                if dz < tau_dzToVtx:
                    tau_dzToVtx = dz
                    tau_vertex_idxpf = i
            '''

            all_var_dict['tau_tauVtxIdx'].fill(tau_vertex_idxpf)
            # Find z distance between tau vertex to other closest vertex
            tau_tauVtxTovtx_dz = 99
            for i, vertex in enumerate(vertices):
                if i == tau_vertex_idxpf:
                    continue

                vtxdz = abs(vertex.z() - vertices[tau_vertex_idxpf].z())
                if vtxdz < tau_tauVtxTovtx_dz:
                    tau_tauVtxTovtx_dz = vtxdz

            all_var_dict['tau_tauVtxTovtx_dz'].fill(tau_tauVtxTovtx_dz)

            all_var_dict['tau_dxy'].fill(tau.dxy())
            all_var_dict['tau_dxy_err'].fill(tau.dxy_error())
            all_var_dict['tau_dxy_sig'].fill(tau.dxy_Sig())
            all_var_dict['tau_ip3d'].fill(tau.ip3d())
            all_var_dict['tau_ip3d_err'].fill(tau.ip3d_error())
            all_var_dict['tau_ip3d_sig'].fill(tau.ip3d_Sig())

            if tau.hasSecondaryVertex():
                all_var_dict['tau_flightLength'].fill(
                    math.sqrt(tau.flightLength().mag2()))
                all_var_dict['tau_flightLength_sig'].fill(
                    tau.flightLengthSig())

            fill_tau_ids(all_var_dict, tau, all_tau_ids)

            ##########################
            
            nPFGamma = 0
            pf_gammas = []
            pi0_reco2 = ROOT.math.XYZTLorentzVectorD()
            pi0_reco3 = ROOT.math.XYZTLorentzVectorD()
            pi0_reco4 = ROOT.math.XYZTLorentzVectorD()
            nEle = 0
            for cand in tau.signalGammaCands(): #MB: consided usage of all gammas around tau (within max strip distance of deta<0.15, dphi<0.3)?
                if ((abs(cand.pdgId()) == 22 
                     or abs(cand.pdgId()) == 11)
                    and cand.pt() > 0.5):
                    if cand.pt()<1 and globaldebug: #MB samity check: signal gammas with pt<1 are not expected 
                        print("pt =",cand.pt(),"< 1!!!, pdgId = ",cand.pdgId())
                    puppi = cand.puppiWeightNoLep()
                    #puppi = cand.puppiWeigh()
                    # strip size w/o considering pt of seed
                    maxEta = max(0.05, min(0.15, 0.20*math.pow(cand.pt(),-0.66)))
                    maxPhi = max(0.05, min(0.30, 0.35*math.pow(cand.pt(),-0.71)))
                    if abs(cand.eta() - tau.eta())<maxEta and abs(deltaPhi(cand.phi(), tau.phi()))<maxPhi:
                        if puppi > puppiMin:
                            nPFGamma += 1
                            pf_gammas.append(cand)
                        if abs(cand.pdgId()) == 11:
                            nEle += 1
            if nPFGamma != len(pf_gammas):
                print('DM:',tau.decayMode(),"#pf-gamma",nPFGamma,len(pf_gammas))
            massLimit = pi0RecoM+3*pi0RecoW #MB <0.2, <0.25?
            ptLimit = 5 #MB <2 <5?
            nPFGammaUsed = 0
            if nPFGamma>=1:
                for iGam in range(0,min(4,nPFGamma)):
                    pi0_reco3 += pf_gammas[iGam].p4()
                    all_var_dict['tau_gammapt'+str(iGam+1)].fill(pf_gammas[iGam].pt())
                    all_var_dict['tau_gammaeta'+str(iGam+1)].fill(pf_gammas[iGam].eta())
                    all_var_dict['tau_gammaphi'+str(iGam+1)].fill(pf_gammas[iGam].phi())
                    all_var_dict['tau_gammadeta'+str(iGam+1)].fill(abs(pf_gammas[iGam].eta()-tau.eta()))
                    all_var_dict['tau_gammadphi'+str(iGam+1)].fill(deltaPhi(pf_gammas[iGam].phi(),tau.phi()))
                    for jGam in range(iGam+1,min(4,nPFGamma)):
                        p4_ij = pf_gammas[iGam].p4()+pf_gammas[jGam].p4()
                        all_var_dict['tau_gammapt'+str(iGam+1)+str(jGam+1)].fill(p4_ij.pt())
                        all_var_dict['tau_gammaeta'+str(iGam+1)+str(jGam+1)].fill(p4_ij.eta())
                        all_var_dict['tau_gammaphi'+str(iGam+1)+str(jGam+1)].fill(p4_ij.phi())
                        all_var_dict['tau_gammadeta'+str(iGam+1)+str(jGam+1)].fill(abs(p4_ij.eta()-tau.eta()))
                        all_var_dict['tau_gammadphi'+str(iGam+1)+str(jGam+1)].fill(deltaPhi(p4_ij.phi(),tau.phi()))
                        all_var_dict['tau_gammamass'+str(iGam+1)+str(jGam+1)].fill(p4_ij.mass())
                        for kGam in range(jGam+1,min(4,nPFGamma)):
                            p4_ijk = pf_gammas[iGam].p4()+pf_gammas[jGam].p4()+pf_gammas[kGam].p4()
                            all_var_dict['tau_gammapt'+str(iGam+1)+str(jGam+1)+str(kGam+1)].fill(p4_ijk.pt())
                            all_var_dict['tau_gammaeta'+str(iGam+1)+str(jGam+1)+str(kGam+1)].fill(p4_ijk.eta())
                            all_var_dict['tau_gammaphi'+str(iGam+1)+str(jGam+1)+str(kGam+1)].fill(p4_ijk.phi())
                            all_var_dict['tau_gammadeta'+str(iGam+1)+str(jGam+1)+str(kGam+1)].fill(abs(p4_ijk.eta()-tau.eta()))
                            all_var_dict['tau_gammadphi'+str(iGam+1)+str(jGam+1)+str(kGam+1)].fill(deltaPhi(p4_ijk.phi(),tau.phi()))
                            all_var_dict['tau_gammamass'+str(iGam+1)+str(jGam+1)+str(kGam+1)].fill(p4_ijk.mass())

                gamma1 = pf_gammas[0]
                pi0_reco2 += gamma1.p4()
                pi0_reco4 += gamma1.p4()
                nPFGammaUsed = 1
                if nPFGamma>=2:
                    gamma2 = pf_gammas[1]
                    deta = abs(gamma1.eta() - gamma2.eta())
                    dphi = abs(deltaPhi(gamma1.phi(), gamma2.phi()))
                    #h_pf_gamma_dist_match.Fill(min(deta,0.2609),min(dphi,0.2609))
                    pi0_reco2 += gamma2.p4()
                    dPhi_1 = abs(deltaPhi(tau.phi(), gamma1.phi()))
                    dPhi_2 = abs(deltaPhi(tau.phi(), gamma2.phi()))
                    if dPhi_1>dPhi_2 or abs(pi0_reco2.mass()-pi0RecoM)<2*pi0RecoW:
                        pi0_reco4 += gamma2.p4()
                        nPFGammaUsed += 1

                # Uncomment to set pi0mass
                '''
                if pi0_reco4.pt()>0:
                    #print ("#pf_gammas",nPFGamma,"pi0 pt=",pi0_reco.pt())
                    scale = math.sqrt(math.pow(pi0_reco4.energy(),2)-math.pow(pi0M,2))/pi0_reco4.P()
                    pi0_reco4.SetXYZT(pi0_reco4.px()*scale,pi0_reco4.py()*scale,pi0_reco4.pz()*scale,pi0_reco4.energy())
                '''

            all_var_dict['tau_nEle'].fill(nEle)
            all_var_dict['tau_nGammas'].fill(nPFGamma)
            all_var_dict['tau_nGammasUsed'].fill(nPFGammaUsed)
            all_var_dict['tau_pi0pt2'].fill(pi0_reco2.pt())
            all_var_dict['tau_pi0eta2'].fill(pi0_reco2.eta())
            all_var_dict['tau_pi0phi2'].fill(pi0_reco2.phi())
            all_var_dict['tau_pi0mass2'].fill(pi0_reco2.mass())
            all_var_dict['tau_pi0pt3'].fill(pi0_reco3.pt())
            all_var_dict['tau_pi0eta3'].fill(pi0_reco3.eta())
            all_var_dict['tau_pi0phi3'].fill(pi0_reco3.phi())
            all_var_dict['tau_pi0mass3'].fill(pi0_reco3.mass())
            all_var_dict['tau_pi0pt4'].fill(pi0_reco4.pt())
            all_var_dict['tau_pi0eta4'].fill(pi0_reco4.eta())
            all_var_dict['tau_pi0phi4'].fill(pi0_reco4.phi())
            all_var_dict['tau_pi0mass4'].fill(pi0_reco4.mass())

            ########################

            tau_tree.Fill()

    print ("MATCHED TAUS:", NMatchedTaus)
    print (evtid, 'events are processed !')

    out_file.Write()
    out_file.Close()
