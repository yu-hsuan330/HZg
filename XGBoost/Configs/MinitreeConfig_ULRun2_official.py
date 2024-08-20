import os 
#===============================================================#
#   All sample information and configuration are listed here    #
#===============================================================#
path = "../../minitrees_OfficialID/"
path_data = "../../minitrees_OfficialID/data/"
branches = ["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1","lepEta1", "lepEta2", "phoEta1", "totwei", "totSF", "mllg", "refit_mllg","phoSCEta1", 'absdPhi_Zgjj', 'dR_phojet', 'absdEta_jj', 'absdPhi_jj', 'VBFPt1', 'VBFPt2', 'absZeppen_pho', 'abssysbal', 'ZgPTt', "VBFtag", "event"] #"ptwei"
features = ["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1","lepEta1", "lepEta2", "phoEta1"]
features_data = ["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1","lepEta1", "lepEta2", "phoEta1"]
Tree = "TH"

samples = [
    {
        "Production": "Signal",
        "Weight": ("totwei","totSF"),
        "Color": "#EF5B5B", 
        "Label": "signal",
        "epath": [path+filename for filename in os.listdir(path) if "ele" in filename and filename.endswith("125GeV.root")],
        "upath": [path+filename for filename in os.listdir(path) if "mu" in filename and filename.endswith("125GeV.root")],
    },
    
    {
        "Production": "SMZg",
        "Weight": ("totwei","totSF"),
        "Color": "#37A3D2",
        "Label": "SM Zg", 
        "epath": [path+filename for filename in os.listdir(path) if filename.endswith("ele_SMZg.root")],
        "upath": [path+filename for filename in os.listdir(path) if filename.endswith("mu_SMZg.root")],
    }, 

    {
        "Production": "DYJets",
        "Weight": ("totwei","totSF"),
        "Color": "#F3C568",
        "Label": "DY+Jets", 
        "epath": [path+filename for filename in os.listdir(path) if filename.endswith("ele_DYJetsToLL.root")],
        "upath": [path+filename for filename in os.listdir(path) if filename.endswith("mu_DYJetsToLL.root")],
    },
    
    {
        "Production": "Data",
        "Weight": ("totwei","totSF"),
        "Color": "#000000",
        "Label": "Data", 
        "epath": ["../../minitrees_OfficialID/data/UL16_preVFP_ele_data*.root", "../../minitrees_OfficialID/data/UL16_postVFP_ele_data*.root", "../../minitrees_OfficialID/data/UL17_ele_data*.root", "../../minitrees_OfficialID/data/UL18_ele_data*.root"],
        "upath": [],

    },
]

MVAlogplot=False