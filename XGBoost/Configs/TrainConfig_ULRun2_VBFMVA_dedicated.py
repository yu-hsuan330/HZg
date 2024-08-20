#=============================================#
#           training configurations           #
#       combined(e+u) channel / UL dataset    #
#               Run-2 variables               #
#=============================================# 
import numpy as np

OutputDirName = "VBFMVA_XGB_ULRun2_dedi" #All plots, models, config file will be stored here
Tree, channel = "TH", "combine"
Debug, MVAlogplot = False, True
testsize, RandomState = 0.5, 123
branches = ["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1", "lepEta1", "lepEta2", "phoEta1", "totwei", "totSF", "ptwei",
            "absdPhi_Zgjj", "dR_phojet", "absdEta_jj", "absdPhi_jj", "VBFPt1", "VBFPt2", "absZeppen_pho", "abssysbal", "ZgPTt",
            "refit_mllgErr", "phoSCEta1"]
features = ["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1", "lepEta1", "lepEta2", "phoEta1",
            "absdPhi_Zgjj", "dR_phojet", "absdEta_jj", "absdPhi_jj", "VBFPt1", "VBFPt2", "absZeppen_pho", "abssysbal", "ZgPTt"]

Classes, ClassColors = ["Signal","Background"], ["#22577E", "#EF5B5B"]
selection = "1"#"((refit_mllg > 105 && refit_mllg < 170.))" #&& event%2 == 1
tag = "(VBFtag == 1)"

processes = [
    {
    "Class":"Signal",
    "path":["../VBFmatch/minitrees_dedicated/UL*_VBF_125GeV_VBFmatch.root"],
    "selection":"((refit_mllg < 170 && refit_mllg > 105) && (VBFtag == 1 && Leptag == 0) && event%2 == 0)",
    "weight": ("totwei", "totSF"),
    },
    {
    "Class":"Background",
    "path":["../../minitrees_dedicated/UL*_DYJetsToLL.root",
            "../../minitrees_dedicated/UL*_SMZg.root",
            "../../minitrees_dedicated/UL*_ggF_125GeV.root"],
    "selection":"((refit_mllg < 170 && refit_mllg > 105) && (VBFtag == 1 && Leptag == 0))", 
    "weight": ("totwei", "totSF"), 
    },
]

MVAs = [
    {
    "MVAtype":"XGB_combine_nowei", 
    "Label":"no-weight",
    "addwei":"nowei",
    "features":["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1", "lepEta1", "lepEta2", "phoEta1",
                "absdPhi_Zgjj", "dR_phojet", "absdEta_jj", "absdPhi_jj", "VBFPt1", "VBFPt2", "absZeppen_pho", "abssysbal", "ZgPTt"],
    "features_unit":["$cos(\\theta)$", "$cos(\\Theta)$", "$\\Phi$", "$p^{T}_{H}/m_{H}$", "photon resol.", "min $\\Delta R(l,\\gamma)$", "max $\\Delta R(l,\\gamma)$","phoID MVA", "$\eta^{l1}$", "$\eta^{l2}$", "$\eta^{\gamma}$",
                     "$|\\Delta \\Phi(Z\\gamma,jj)|$", "min $\\Delta R(\\gamma, j)$", "$|\\Delta \\eta(j,j)|$", "$|\\Delta \\Phi(j,j)|$", "$p^{T}_{j1}$", "$p^{T}_{j2}$", "$|\\gamma Zeppend|$", "system balance", "$Z\\gamma p^{T}_{t}$"],
    "feature_bins":[30,30,25,np.linspace(0, 2.5, 31),np.linspace(0, 0.3, 31),30,30,30,30,30,30,30,30,30,30,np.linspace(25, 300, 31),np.linspace(25, 150, 31),np.linspace(30, 300, 31),np.linspace(25, 125, 31),30],
    "hyperopt":False,
    "ModelParams":{"objective":"binary:logistic","eval_metric":"auc","tree_method":"gpu_hist","random_state":RandomState},
    "HyperParams":{"learning_rate":0.09961607168398715, "max_delta_step":12, "max_depth": 3,"min_child_weight": 6.222294568754226, "min_split_loss": 0.3449815537541523, "subsample": 0.31997385915947707},
    },
    {
    "MVAtype":"XGB_combine_refit_mllgErr",
    "Label":"refit mllgErr",
    "addwei":"refit_mllgErr",
    "features":["costheta","cosTheta","Phi","mllgptdmllg","phores","dR_lg","maxdR_lg","phoCalIDMVA1","lepEta1","lepEta2","phoEta1",
                "absdPhi_Zgjj", "dR_phojet", "absdEta_jj", "absdPhi_jj", "VBFPt1", "VBFPt2", "absZeppen_pho", "abssysbal", "ZgPTt"],
    "features_unit":["$cos(\\theta)$","$cos(\\Theta)$","$\\Phi$","$p^{T}_{H}/m_{H}$","photon resol.","min $\\Delta R(l,\\gamma)$","max $\\Delta R(l,\\gamma)$","phoID MVA","$\eta^{l1}$","$\eta^{l2}$","$\eta^{\gamma}$",
                     "$|\\Delta \\Phi(Z\\gamma,jj)|$", "min $\\Delta R(\\gamma, j)$", "$|\\Delta \\eta(j,j)|$", "$|\\Delta \\Phi(j,j)|$", "$p^{T}_{j1}$", "$p^{T}_{j2}$", "$|\\gamma Zeppend|$", "system balance", "$Z\\gamma p^{T}_{t}$"],
    "feature_bins":[30,30,25,np.linspace(0, 2.5, 31),np.linspace(0, 0.3, 31),30,30,30,30,30,30,30,30,30,30,np.linspace(0, 350, 31),np.linspace(25, 160, 31),np.linspace(30, 300, 31),np.linspace(25, 150, 31),30],
    "hyperopt":False,
    "ModelParams":{"objective":"binary:logistic","eval_metric":"auc","tree_method":"gpu_hist","random_state":RandomState},
    "HyperParams":{"learning_rate": 0.0926099229150509, "max_delta_step":8, "max_depth": 3,"min_child_weight":8.615376070911601 , "min_split_loss": 1.1021533745112295, "subsample": 0.5771096236932081},
    },
]