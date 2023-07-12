basic_tau_ids = [
    ('byCombinedIsolationDeltaBetaCorrRaw3Hits', float),
    ('byLooseCombinedIsolationDeltaBetaCorr3Hits', int),
    ('byMediumCombinedIsolationDeltaBetaCorr3Hits', int),
    ('byTightCombinedIsolationDeltaBetaCorr3Hits', int),

    ('chargedIsoPtSum', float),
    ('neutralIsoPtSum', float),
    ('puCorrPtSum', float),
    ('neutralIsoPtSumWeight', float),
    ('footprintCorrection', float),
    ('photonPtSumOutsideSignalCone', float),
    ('decayModeFinding', int),
    ('decayModeFindingNewDMs', int),
]

mvaiso_tau_ids = [
    ('byIsolationMVArun2v1DBoldDMwLTraw', float),
    ('byVVLooseIsolationMVArun2v1DBoldDMwLT', int),
    ('byVLooseIsolationMVArun2v1DBoldDMwLT', int),
    ('byLooseIsolationMVArun2v1DBoldDMwLT', int),
    ('byMediumIsolationMVArun2v1DBoldDMwLT', int),
    ('byTightIsolationMVArun2v1DBoldDMwLT', int),
    ('byVTightIsolationMVArun2v1DBoldDMwLT', int),
    ('byVVTightIsolationMVArun2v1DBoldDMwLT', int),

    #('byIsolationMVArun2v1PWoldDMwLTraw', float), # Gone? Not in 11_0_0_pre13
    #('byVLooseIsolationMVArun2v1PWoldDMwLT', int),
    #('byLooseIsolationMVArun2v1PWoldDMwLT', int),
    #('byMediumIsolationMVArun2v1PWoldDMwLT', int),
    #('byTightIsolationMVArun2v1PWoldDMwLT', int),
    #('byVTightIsolationMVArun2v1PWoldDMwLT', int),
    #('byVVTightIsolationMVArun2v1PWoldDMwLT', int),

    ('byIsolationMVArun2v1DBnewDMwLTraw', float),
    ('byVVLooseIsolationMVArun2v1DBnewDMwLT', int),
    ('byVLooseIsolationMVArun2v1DBnewDMwLT', int),
    ('byLooseIsolationMVArun2v1DBnewDMwLT', int),
    ('byMediumIsolationMVArun2v1DBnewDMwLT', int),
    ('byTightIsolationMVArun2v1DBnewDMwLT', int),
    ('byVTightIsolationMVArun2v1DBnewDMwLT', int),
    ('byVVTightIsolationMVArun2v1DBnewDMwLT', int),

    ('byIsolationMVArun2v1DBdR03oldDMwLTraw', float),
    ('byVVLooseIsolationMVArun2v1DBdR03oldDMwLT', int),
    ('byVLooseIsolationMVArun2v1DBdR03oldDMwLT', int),
    ('byLooseIsolationMVArun2v1DBdR03oldDMwLT', int),
    ('byMediumIsolationMVArun2v1DBdR03oldDMwLT', int),
    ('byTightIsolationMVArun2v1DBdR03oldDMwLT', int),
    ('byVTightIsolationMVArun2v1DBdR03oldDMwLT', int),
    ('byVVTightIsolationMVArun2v1DBdR03oldDMwLT', int),
]

lepton_tau_ids = [
    ('againstMuonLoose3', int),
    ('againstMuonTight3', int),
    ('againstElectronVLooseMVA6', int),
    ('againstElectronLooseMVA6', int),
    ('againstElectronMediumMVA6', int),
    ('againstElectronTightMVA6', int),
    ('againstElectronVTightMVA6', int),
    ('againstElectronMVA6Raw', float),
]

all_wps = ['VVLoose', 'VLoose', 'Loose', 'Medium', 'Tight', 'VTight', 'VVTight']

def create_tau_ids(name, n_wps=7):
    if name=="againstMuon3":
        return [('againstMuonLoose3', int),
                ('againstMuonTight3', int)]
    elif name=="MVADM":
        return [("MVADM2017v1", float)]
        
    wps = all_wps[:]
    if n_wps == 6:
        wps = all_wps[1:]
    if n_wps == 5:
        wps = all_wps[1:6]
    if n_wps == 8:
        wps.append('VVVLoose')
    if n_wps == 4:
        wps = all_wps[1:5]
    if name[-4:-1] == "201":
      rawname = 'by' + name[:-4] + 'raw' + name[-4:]
    else:
      rawname = 'by' + name + 'raw'
    return [('by' + wp + name, int) for wp in wps] + [(rawname, float)]


tau_ids = {
    'deepTau2018v2p5VSe':create_tau_ids(name='DeepTau2018v2p5VSe', n_wps=8),
    'deepTau2018v2p5VSmu':create_tau_ids(name='DeepTau2018v2p5VSmu', n_wps=4),
    'deepTau2018v2p5VSjet':create_tau_ids(name='DeepTau2018v2p5VSjet', n_wps=8),
    'deepTau2017v2p1VSe':create_tau_ids(name='DeepTau2017v2p1VSe', n_wps=8),
    'deepTau2017v2p1VSmu':create_tau_ids(name='DeepTau2017v2p1VSmu', n_wps=4),
    'deepTau2017v2p1VSjet':create_tau_ids(name='DeepTau2017v2p1VSjet', n_wps=8),
    '2017v2':create_tau_ids('IsolationMVArun2017v2DBoldDMwLT2017'),
    '2017v1':create_tau_ids('IsolationMVArun2017v1DBoldDMwLT2017'),
    '2016v1':create_tau_ids('IsolationMVArun2v1DBoldDMwLT2016', n_wps=6),
    'newDM2016v1':create_tau_ids('IsolationMVArun2v1DBnewDMwLT2016', n_wps=6),
    'dR0p32017v2':create_tau_ids('IsolationMVArun2017v2DBoldDMdR0p3wLT2017'),

    'againstMuon3':create_tau_ids('againstMuon3'),
    'MVADM':create_tau_ids('MVADM'),
}

def fill_tau_ids(avd, tau, tau_id_names):
    for (tau_id, _) in tau_id_names:
        if tau.isTauIDAvailable(tau_id):
            avd['tau_'+tau_id].fill(tau.tauID(tau_id))
        else:
            avd['tau_'+tau_id].fill(-999)
