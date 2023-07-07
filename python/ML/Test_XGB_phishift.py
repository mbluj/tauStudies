##
## \date August 2022
## \author Michal Bluj
##

import ROOT
from ROOT import gROOT, gStyle, TH1F, TH2F, TCanvas, TLegend
from officialStyle import officialStyle

import os
import numpy as np
import pickle
import math

from DataPreparation import load_data
import utility_functions as utils

def configureLegend(leg, ncolumn):
    leg.SetNColumns(ncolumn)
    leg.SetBorderSize(0)
    leg.SetFillColor(10)
    leg.SetLineColor(0)
    leg.SetFillStyle(0)
    #leg.SetTextSize(0.02)
    leg.SetTextSize(0.035)
    leg.SetTextFont(42)

if __name__ == "__main__":
    # Load test data
    #reduced = False
    reduced = True

    target = 'phi'
    #target = 'eta'
    #target = 'pt'

    training_type = 'signal'
    #training_type = 'tauGun'
    #x_t, y_t, z_t = load_data('test_signal.root',target=target,reduced=reduced)
    x_t, y_t, z_t = load_data('test_signal_deep2p5.root',target=target,reduced=reduced)
    model_dir = "training/"
    plot_dir = "figures/"
    os.makedirs(plot_dir, exist_ok=True)

    # Load trained model
    from xgboost import XGBRegressor
    bdt = XGBRegressor()
    model_name = ""
    if not reduced:
        model_name = utils.getLatestModelPath(trainingPath=model_dir, pattern="model_shift_"+training_type+"_"+target)
    else:
        model_name = utils.getLatestModelPath(trainingPath=model_dir, pattern="model_shift_reduced_"+training_type+"_"+target)
    model_pattern = model_name[:model_name.find(".json")]

    print("Loading model:", model_name, flush=True)
    bdt.load_model(model_dir+model_name)

    y_test = None
    if target!='phi': #gen phi is shifted
        y_test = y_t[:,0] #get 0th column (only one stored anyway) 
    else:
        y_test = z_t[:,8] #last column here is no-shifted gen phi
    z_test = None
    z2_test = None
    shift = None
    if target=='phi':
        z_test = z_t[:,2] #get 2nd column (phi): pt/eta/phi/mass
        z2_test = z_t[:,6] #get (4+2)th column (2nd phi): pt/eta/phi
        shift = z_t[:,7] #
    elif target=='eta':
        z_test = z_t[:,1] #get 1st column (eta): pt/eta/phi/mass
        z2_test = z_t[:,5] #get (4+1)th column (2nd eta): pt/eta/phi
    elif target=='pt':
        z_test = z_t[:,0] #get 0th column (pt): pt/eta/phi/mass
        z2_test = z_t[:,4] #get (4+0)th column (2nd pt): pt/eta/phi

    y_p = bdt.predict(x_t)
    y_pred = y_p[:]
    
    print("Score:", bdt.score(x_t,y_t), flush=True)

    gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    officialStyle(gStyle)
    gStyle.SetOptTitle(0)

    c = TCanvas("c", "", 600, 600)
    h = None
    h2 = None
    h3 = None
    h_2d = None
    h2_2d = None
    h3_2d = None
    if target=='phi':
        h = TH1F("h",";#phi(#pi^{0})_{reco}- #phi(#pi^{0})_{gen} (rad)",200,-0.5,0.5)
        h2 = TH1F("h2",";#phi(#pi^{0})_{reco}- #phi(#pi^{0})_{gen} (rad)",200,-0.5,0.5)
        h3 = TH1F("h3",";#phi(#pi^{0})_{reco}- #phi(#pi^{0})_{gen} (rad)",200,-0.5,0.5)
        h_2d = TH2F("h_2d",";#phi(#pi^{0})_{gen} (rad);#phi(#pi^{0})_{XGB} (rad)",700,-3.5,3.5,700,-3.5,3.5)
        h2_2d = TH2F("h2_2d",";#phi(#pi^{0})_{gen} (rad);#phi(#pi^{0})_{my} (rad)",700,-3.5,3.5,700,-3.5,3.5)
        h3_2d = TH2F("h3_2d",";#phi(#pi^{0})_{gen} (rad);#phi(#pi^{0})_{lead-#gamma} (rad)",700,-3.5,3.5,700,-3.5,3.5)
    elif target=='eta':
        h = TH1F("h",";#eta(#pi^{0})_{reco}- #eta(#pi^{0})_{gen}",120,-0.3,0.3)
        h2 = TH1F("h2",";#eta(#pi^{0})_{reco}- #eta(#pi^{0})_{gen}",120,-0.3,0.3)
        h3 = TH1F("h3",";#eta(#pi^{0})_{reco}- #eta(#pi^{0})_{gen}",120,-0.3,0.3)
        h_2d = TH2F("h_2d",";#eta(#pi^{0})_{gen};#eta(#pi^{0})_{XGB}",600,-3.,3.,600,-3.,3.)
        h2_2d = TH2F("h2_2d",";#eta(#pi^{0})_{gen};#eta(#pi^{0})_{my}",600,-3.,3.,600,-3.,3.)
        h3_2d = TH2F("h3_2d",";#eta(#pi^{0})_{gen};#eta(#pi^{0})_{lead-#gamma}",600,-3.,3.,600,-3.,3.)
    elif target=='pt':
        h = TH1F("h",";p_{T}(#pi^{0})_{reco}/ p_{T}(#pi^{0})_{gen}- 1",250,-1.,1.5)
        h2 = TH1F("h2",";p_{T}(#pi^{0})_{reco}/ p_{T}(#pi^{0})_{gen}- 1",250,-1.,1.5)
        h3 = TH1F("h3",";p_{T}(#pi^{0})_{reco}/ p_{T}(#pi^{0})_{gen}- 1",250,-1.,1.5)
        h_2d = TH2F("h_2d","p_{T}(#pi^{0})_{gen} (GeV);p_{T}(#pi^{0})_{XGB} (GeV)",200,0.,100.,200,0.,100.)
        h2_2d = TH2F("h2_2d",";p_{T}(#pi^{0})_{gen} (GeV);p_{T}(#pi^{0})_{my} (GeV)",200,0.,100,200,0.,100.)
        h3_2d = TH2F("h3_2d",";p_{T}(#pi^{0})_{gen} (GeV);p_{T}(#pi^{0})_{lead-#gamma} (GeV)",200,0.,100.,200,0.,100.)

    h.StatOverflows(True)
    h.Sumw2()
    h2.StatOverflows(True)
    h2.Sumw2()
    h3.StatOverflows(True)
    h3.Sumw2()
    
    mae = 0
    mse = 0
    N = len(y_test)
    for i in range(0,N):
        diff = 0
        if target=='phi':
            diff = utils.phi_mpi_pi(utils.phi_mpi_pi(y_pred[i]+shift[i])-y_test[i])
            diff2 = utils.phi_mpi_pi(z_test[i]-y_test[i])
            diff3 = utils.phi_mpi_pi(z2_test[i]-y_test[i])
        elif target=='eta':
            diff = y_pred[i]-y_test[i]
            diff2 = z_test[i]-y_test[i]
            diff3 = z2_test[i]-y_test[i]
        elif target=='pt':
            if y_test[i]>0: #should be always true
                diff = y_pred[i]/y_test[i]-1
                diff2 = z_test[i]/y_test[i]-1
                diff3 = z2_test[i]/y_test[i]-1
        mae += abs(diff) # Does it make sense for (?)
        mse += math.pow(diff,2) # Does it make sense for (?)
        h.Fill(diff)
        h2.Fill(diff2)
        h3.Fill(diff3)
        if target=='phi':
            h_2d.Fill(y_test[i],utils.phi_mpi_pi(y_pred[i]+shift[i]))
        else:
            h_2d.Fill(y_test[i],y_pred[i])
        h2_2d.Fill(y_test[i],z_test[i])
        h3_2d.Fill(y_test[i],z2_test[i])
    mae = mae / N
    mse = mse / N
    rmse = math.sqrt(mse)
    h.SetLineWidth(2)
    h.SetLineColor(ROOT.kRed)
    h2.SetLineWidth(2)
    h2.SetLineColor(ROOT.kBlue)
    h3.SetLineWidth(2)
    h3.SetLineColor(ROOT.kGreen+2)
    firstToPlot = h
    if h2.GetBinContent(h2.GetMaximumBin()) > firstToPlot.GetBinContent(firstToPlot.GetMaximumBin()):
        firstToPlot = h2
    if h3.GetBinContent(h2.GetMaximumBin()) > firstToPlot.GetBinContent(firstToPlot.GetMaximumBin()):
        firstToPlot = h3
    firstToPlot.Draw("axis")
    firstToPlot.Draw("axig same")
    h.Draw("same hist")
    h2.Draw("same hist")
    h3.Draw("same hist")
    leg = None
    if target!='pt':
        leg = TLegend(0.6, 0.75, 0.95, 0.9)
    else:
        leg = TLegend(0.5, 0.75, 0.75, 0.9)
    configureLegend(leg, 1)
    mean = f'{h.GetMean(1):.2f}'
    sigma = f'{h.GetStdDev(1):.3f}'
    #sigma = f'{h.GetStdDev(1):.7f}'
    print('target: {}, N = {}, sigma = {}, rmse = {:.3f}, mae = {:.3f}'.format(target,N,sigma,rmse,mae))
    mean2 = f'{h2.GetMean(1):.2f}'
    sigma2 = f'{h2.GetStdDev(1):.3f}'
    mean3 = f'{h3.GetMean(1):.2f}'
    sigma3 = f'{h3.GetStdDev(1):.3f}'
    if target!='pt':
        leg.AddEntry(h, "XGB (#sigma="+str(sigma)+")", "l")
        leg.AddEntry(h2, "my #pi^{0} (#sigma="+str(sigma2)+")", "l")
        leg.AddEntry(h3, "#gamma^{lead}_{PF} (#sigma="+str(sigma3)+")", "l")
    else:
        leg.AddEntry(h, "XGB (#mu="+str(mean)+", #sigma="+str(sigma)+")", "l")
        leg.AddEntry(h2, "my #pi^{0} (#mu="+str(mean2)+", #sigma="+str(sigma2)+")", "l")
        leg.AddEntry(h3, "#gamma^{lead}_{PF} (#mu="+str(mean3)+", #sigma="+str(sigma3)+")", "l")
    leg.Draw()
    #c.SetGrid()

    c.SetLogy(1)
    c.Draw()
    c.Print(plot_dir+model_pattern+'_log.png')

    c.SetLogy(0)
    if target=='phi':
        firstToPlot.GetXaxis().SetRangeUser(-0.1,0.1)
    elif target=='eta':
        firstToPlot.GetXaxis().SetRangeUser(-0.05,0.05)
    firstToPlot.Draw("axis")
    firstToPlot.Draw("axig same")
    h.Draw("same hist")
    h2.Draw("same hist")
    h3.Draw("same hist")
    leg.Draw()

    c.Draw()
    c.Print(plot_dir+model_pattern+'.png')

    c.Clear()
    c.Divide(2,2)
    c.cd(1).SetLogz(1)
    h_2d.Draw("col z")
    c.cd(3).SetLogz(1)
    h2_2d.Draw("col z")
    c.cd(4).SetLogz(1)
    h3_2d.Draw("col z")
    c.Print(plot_dir+model_pattern+'_2d.png')
