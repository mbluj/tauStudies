##
## \date August 2022
## \author Michal Bluj
## based on $ROOTSYS/tutorials/tmva/tmva100_DataPreparation.py by Stefan Wunsch
##

import ROOT
import copy
import numpy as np

########################
def filter_events(df, dm=1):
    """
    Reduce initial dataset to only events which shall be used for training
    """
    selStr = "tau_genmatch==5" # gen matching
    if dm>=0:
        selStr += " && tau_gendm=="+str(dm) # select gen decay mode
    selStr += " && tau_nGammas>1" # at least two pf-gammas to estimate pi0 momentum (with one gamma estimation is trivial)
    selStr += " && tau_dm<5" # 1-prong
    selStr += " && abs(tau_dzPV)<0.2" # from PV
    # add loose deepTau selection? loosestWP vsj, e, mu? or of v2p1 and v2p5??
    return df.Filter(selStr, "loosely selected reco-tau matched to gen with gen decay mode = "+str(dm)+" and at least one associated pf-gamma")


########################
def define_variables(df):
    """
    Define the variables which shall be used for training
    MB: only needed for renamed ones (including elements of tables)?
    """
    return df.Define("tau_mvadm", "tau_MVADM2017v1")\
             .Define("tau_genpi0pt", "tau_genneutralpt")\
             .Define("tau_genpi0eta", "tau_genneutraleta")\
             .Define("tau_genpi0phi", "tau_genneutralphi")\
             .Define("tau_genmatch", "tau_match")\
             .Define("rho", "tau_rho")\
             .Define("tau_deeptauv2p1vseraw","tau_byDeepTau2017v2p1VSeraw")\
             .Define("tau_deeptauv2p1vsmuraw","tau_byDeepTau2017v2p1VSmuraw")\
             .Define("tau_deeptauv2p1vsjetraw","tau_byDeepTau2017v2p1VSjetraw")\
             .Define("tau_deeptauv2p5vseraw","tau_byDeepTau2018v2p5VSeraw")\
             .Define("tau_deeptauv2p5vsmuraw","tau_byDeepTau2018v2p5VSmuraw")\
             .Define("tau_deeptauv2p5vsjetraw","tau_byDeepTau2018v2p5VSjetraw")\
             .Define("event", "tau_eventid")


########################
variables = []
variables += ["tau_pt", "tau_eta", "tau_phi"]
variables += ["tau_neutralpt", "tau_neutraleta", "tau_neutralphi"]
variables += ["tau_pi0pt2", "tau_pi0eta2", "tau_pi0phi2","tau_pi0mass2"] #pi0 with up to two leading gammas
variables += ["tau_pi0pt3", "tau_pi0eta3", "tau_pi0phi3","tau_pi0mass3"] #pi0 with all gammas in "strip2"; close to tau_neutral{pt,eta,phi} (pi0 with all signal gammas), needed?
#individual gammas and their combinations (up to 4)
for iGam in range(0,4):
    variables.append('tau_gammapt'+str(iGam+1))
    variables.append('tau_gammaeta'+str(iGam+1))
    variables.append('tau_gammaphi'+str(iGam+1))
    variables.append('tau_gammadeta'+str(iGam+1))
    variables.append('tau_gammadphi'+str(iGam+1))
    for jGam in range(iGam+1,4): #pairs
        variables.append('tau_gammapt'+str(iGam+1)+str(jGam+1))
        variables.append('tau_gammaeta'+str(iGam+1)+str(jGam+1))
        variables.append('tau_gammaphi'+str(iGam+1)+str(jGam+1))
        variables.append('tau_gammadeta'+str(iGam+1)+str(jGam+1))
        variables.append('tau_gammadphi'+str(iGam+1)+str(jGam+1))
        variables.append('tau_gammamass'+str(iGam+1)+str(jGam+1))
        for kGam in range(jGam+1,4): #triplets
            variables.append('tau_gammapt'+str(iGam+1)+str(jGam+1)+str(kGam+1))
            variables.append('tau_gammaeta'+str(iGam+1)+str(jGam+1)+str(kGam+1))
            variables.append('tau_gammaphi'+str(iGam+1)+str(jGam+1)+str(kGam+1))
            variables.append('tau_gammadeta'+str(iGam+1)+str(jGam+1)+str(kGam+1))
            variables.append('tau_gammadphi'+str(iGam+1)+str(jGam+1)+str(kGam+1))
            variables.append('tau_gammamass'+str(iGam+1)+str(jGam+1)+str(kGam+1))
variables += ['tau_nGammas','tau_nEle']
variables += ["tau_dm", "tau_mvadm"]
variables += ["rho"]

targetVars = ["tau_genpi0pt", "tau_genpi0eta", "tau_genpi0phi"]

otherVars = ["tau_gendm", "tau_genmatch", "event"]
otherVars += ["tau_deeptauv2p1vseraw","tau_deeptauv2p1vsmuraw","tau_deeptauv2p1vsjetraw"]
otherVars += ["tau_deeptauv2p5vseraw","tau_deeptauv2p5vsmuraw","tau_deeptauv2p5vsjetraw"]
bestGuess = ["tau_pi0pt4", "tau_pi0eta4", "tau_pi0phi4","tau_pi0mass4"]

allVars = variables + otherVars + bestGuess + targetVars

########################
def load_data(filename, target='phi', reduced=False):

    if target not in ['phi','eta','pt']:
        print("WARNING! Incorrect target \"{}\" - default will be used, i.e. \"phi\"".format(target))
        target = 'phi'

    pi0RecoM = 0.136 #approximate pi0 mass taken from early fits do data in PF paper
    # Read data from ROOT files
    data_sig = ROOT.RDataFrame("per_tau", filename)\
                   .Define("tau_gammadphi1_abs", "(tau_gammapt1>0)*abs(tau_gammadphi1)")\
                   .Define("tau_gammadphi2_abs", "(tau_gammapt2>0)*abs(tau_gammadphi2)")\
                   .Define("tau_gammadphi3_abs", "(tau_gammapt3>0)*abs(tau_gammadphi3)")\
                   .Define("tau_gammadphi4_abs", "(tau_gammapt4>0)*abs(tau_gammadphi4)")\
                   .Define("tau_gammamass12_diff", "(tau_gammapt2>0)*abs(tau_gammamass12-{})-1*(tau_gammapt2<0)".format(pi0RecoM))\
                   .Define("tau_gammamass13_diff", "(tau_gammapt3>0)*abs(tau_gammamass13-{})-1*(tau_gammapt3<0)".format(pi0RecoM))\
                   .Define("tau_gammamass14_diff", "(tau_gammapt4>0)*abs(tau_gammamass14-{})-1*(tau_gammapt4<0)".format(pi0RecoM))\
                   .Define("tau_gammamass23_diff", "(tau_gammapt3>0)*abs(tau_gammamass23-{})-1*(tau_gammapt3<0)".format(pi0RecoM))\
                   .Define("tau_gammamass24_diff", "(tau_gammapt4>0)*abs(tau_gammamass24-{})-1*(tau_gammapt4<0)".format(pi0RecoM))\
                   .Define("tau_gammamass34_diff", "(tau_gammapt4>0)*abs(tau_gammamass34-{})-1*(tau_gammapt4<0)".format(pi0RecoM))\
                   .Define("tau_pi0mass3_diff", "abs(tau_pi0mass3-{})".format(pi0RecoM))\
                   .Define("tau_pi0mass4_diff", "abs(tau_pi0mass4-{})".format(pi0RecoM))\
                   .Define("tau_gammaphi1_shift","TVector2::Phi_mpi_pi(tau_gammaphi1-tau_phi)")\
                   .Define("tau_gammaphi2_shift","TVector2::Phi_mpi_pi(tau_gammaphi2-tau_phi)")\
                   .Define("tau_gammaphi12_shift","TVector2::Phi_mpi_pi(tau_gammaphi12-tau_phi)")\
                   .Define("tau_gammaphi3_shift","TVector2::Phi_mpi_pi(tau_gammaphi3-tau_phi)*(tau_gammapt3>0)")\
                   .Define("tau_gammaphi4_shift","TVector2::Phi_mpi_pi(tau_gammaphi4-tau_phi)*(tau_gammapt4>0)")\
                   .Define("tau_gammaphi13_shift","TVector2::Phi_mpi_pi(tau_gammaphi13-tau_phi)*(tau_gammapt3>0)")\
                   .Define("tau_gammaphi23_shift","TVector2::Phi_mpi_pi(tau_gammaphi23-tau_phi)*(tau_gammapt3>0)")\
                   .Define("tau_gammaphi14_shift","TVector2::Phi_mpi_pi(tau_gammaphi14-tau_phi)*(tau_gammapt4>0)")\
                   .Define("tau_gammaphi24_shift","TVector2::Phi_mpi_pi(tau_gammaphi24-tau_phi)*(tau_gammapt4>0)")\
                   .Define("tau_gammaphi34_shift","TVector2::Phi_mpi_pi(tau_gammaphi34-tau_phi)*(tau_gammapt4>0)")\
                   .Define("tau_pi0phi3_shift","TVector2::Phi_mpi_pi(tau_pi0phi3-tau_phi)")\
                   .Define("tau_pi0phi4_shift","TVector2::Phi_mpi_pi(tau_pi0phi4-tau_phi)")\
                   .Define("tau_neutralphi_shift","TVector2::Phi_mpi_pi(tau_neutralphi-tau_phi)")\
                   .Define("tau_genpi0phi_shift","TVector2::Phi_mpi_pi(tau_genpi0phi-tau_phi)")\
                   .Define("tau_gammaeta3_cor","tau_gammaeta3*(tau_gammapt3>0)+tau_eta*(tau_gammapt3<0)")\
                   .Define("tau_gammaeta4_cor","tau_gammaeta4*(tau_gammapt4>0)+tau_eta*(tau_gammapt4<0)")\
                   .Define("tau_gammadeta3_cor","tau_gammadeta3*(tau_gammapt3>0)")\
                   .Define("tau_gammadeta4_cor","tau_gammadeta4*(tau_gammapt4>0)")\
                   .AsNumpy()

    myVars = [
        'tau_gammapt1', 'tau_gammaeta1','tau_gammaphi1_shift','tau_gammadphi1_abs','tau_gammadeta1',
        'tau_gammapt2', 'tau_gammaeta2','tau_gammaphi2_shift','tau_gammadphi2_abs','tau_gammadeta2',
        'tau_gammapt12', 'tau_gammaeta12','tau_gammaphi12_shift','tau_gammamass12_diff',
        'tau_nGammas'
    ]
    myVars += ["tau_neutralpt", "tau_neutraleta", "tau_neutralphi_shift"]
    #myVars += ["tau_pi0pt4", "tau_pi0eta4", "tau_pi0phi4_shift","tau_pi0mass4"]#my best guess
    if not reduced:
        #myVars += copy.deepcopy(variables)
        myVars += [
            'tau_pi0mass3_diff',
            'tau_gammapt3', 'tau_gammaeta3_cor','tau_gammaphi3_shift','tau_gammadphi3_abs','tau_gammadeta3_cor',
            'tau_gammapt4', 'tau_gammaeta4_cor','tau_gammaphi4_shift','tau_gammadphi4_abs','tau_gammadeta4_cor',
            'tau_gammapt13', 'tau_gammaeta13','tau_gammaphi13_shift','tau_gammamass13_diff',
            'tau_gammapt14', 'tau_gammaeta14','tau_gammaphi14_shift','tau_gammamass14_diff',
            'tau_gammapt23', 'tau_gammaeta23','tau_gammaphi23_shift','tau_gammamass23_diff',
            'tau_gammapt34', 'tau_gammaeta34','tau_gammaphi34_shift','tau_gammamass34_diff',
            'tau_dm','tau_mvadm','rho','tau_nEle'
        ]
    # Convert inputs to format readable by machine learning tools
    x = np.vstack([data_sig[var] for var in myVars]).T
    #myTargetVars = copy.deepcopy(targetVars) # not possible to define several targets in one go with available XGB version
    myTargetVars = []
    if target=='phi':
        #myTargetVars = ["tau_genpi0phi"]
        myTargetVars = ["tau_genpi0phi_shift"]
    elif target=='eta':
        myTargetVars = ["tau_genpi0eta"]
    elif target=='pt':
        myTargetVars = ["tau_genpi0pt"]
    y = np.vstack([data_sig[var] for var in myTargetVars]).T

    # my best estimate
    myBS = copy.deepcopy(bestGuess)
    myBS += ['tau_gammapt1','tau_gammaeta1','tau_gammaphi1']
    myBS += ['tau_phi','tau_genpi0phi']
    z = np.vstack([data_sig[var] for var in myBS]).T

    return x, y, z #myVars, myTargetVars, myBS

################################################
if __name__ == "__main__":
    for filename, label in [["data/ggH/Myroot_XYZ_ABC_ggHTT*.root", "signal"], ]:
    #for filename, label in [["data/tauGun/Myroot_XYZ_ABC_tauGun*.root", "tauGun"], ]:
        print(">>> Extract the training and testing events for {} from the {} dataset."
              .format(label, filename))

        # Load dataset, filter the required events and define the training variables
        #filepath = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/" + filename
        filepath = "file:./" + filename
        ROOT.EnableImplicitMT();
        #df = ROOT.RDataFrame("Events", filepath)
        df = ROOT.RDataFrame("per_tau", filepath)
        df = define_variables(df)
        df = filter_events(df,dm=1)

        # Book cutflow report
        report = df.Report()

        # Split dataset by event number for training and testing
        columns = ROOT.std.vector["string"](allVars)
        df.Filter("event % 3 == 1", "Select events with event number %3==1 for training")\
          .Snapshot("per_tau", "train_" + label + ".root", columns)
        df.Filter("event % 3 == 2", "Select events with event number %3==2 for testing")\
          .Snapshot("per_tau", "test_" + label + ".root", columns)
        df.Filter("event % 3 == 2 && (tau_deeptauv2p5vsjetraw > 0.9632 && tau_deeptauv2p5vsmuraw > 0.2949 && tau_deeptauv2p5vseraw > 0.0990)", "Select events with event number %3==2 and passing looses deepTau v2.5 WPs (MvsJet, VLvsMu, VVVLvsE ~65%) for testing")\
          .Snapshot("per_tau", "test_" + label + "_deep2p5.root", columns)
        df.Filter("event % 3 == 0", "Select events with event number divisible by three (%3==0) for validation")\
          .Snapshot("per_tau", "validate_" + label + ".root", columns)

        # Print cutflow report
        report.Print()
