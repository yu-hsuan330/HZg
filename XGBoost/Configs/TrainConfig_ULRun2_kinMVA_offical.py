#=============================================#
#           training configurations           #
#       combined(e+u) channel / UL dataset    #
#               Run-2 variables               #
#=============================================# 
import numpy as np

OutputDirName = "kinMVA_XGB_ULRun2_Offi" #All plots, models, config file will be stored here
Tree, channel = "TH", "combine"
Debug, MVAlogplot = False, False
testsize, RandomState = 0.5, 89

branches = ["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1", "lepEta1", "lepEta2", "phoEta1", 
            "totwei", "totSF", "refit_mllgErr", "phoSCEta1", "event"]
features = ["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1", "lepEta1", "lepEta2", "phoEta1"]

Classes, ClassColors = ["Signal","Background"], ["#22577E", "#EF5B5B"]
selection = "1"
tag = "(VBFtag+Leptag == 0)"
processes = [
    {
    "Class":"Signal",
    "path":["../minitrees_official/UL*_125GeV.root"],
    "selection":"(refit_mllg < 170) & (refit_mllg > 105) & (VBFtag+Leptag == 0)",
    "weight": ("totwei", "totSF"),
    },
    
    {
    "Class":"Background",
    "path":["../minitrees_official/UL*_DYJetsToLL.root",
            "../minitrees_official/UL*_SMZg.root"],
    "selection":"(refit_mllg < 170) & (refit_mllg > 105) & (VBFtag+Leptag == 0)", 
    "weight": ("totwei", "totSF"),
    },
]

MVAs = [
    {
    "MVAtype":"XGB_combine_nowei", 
    "Label":"no-weight",
    "addwei":"nowei",
    "features":["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1", "lepEta1", "lepEta2", "phoEta1"],
    "features_unit":["$cos(\\theta)$", "$cos(\\Theta)$", "$\\Phi$", "$p^{T}_{H}/m_{H}$", "photon resol.", "min $\\Delta R(l,\\gamma)$", "max $\\Delta R(l,\\gamma)$",
                    "phoID MVA", "$\eta^{l1}$", "$\eta^{l2}$", "$\eta^{\gamma}$"],
    "feature_bins":[30,30,25,np.linspace(0, 2.5, 31),np.linspace(0, 0.3, 31),30,30,30,30,30,30],
    "hyperopt":True,
    "ModelParams":{"objective":"binary:logistic","eval_metric":"auc","tree_method":"gpu_hist","random_state":RandomState},
    "HyperParams":{"learning_rate": 0.2, "max_delta_step": 12, "max_depth": 7,"min_child_weight": 604.1819683416226, "min_split_loss": 2.08820497814834, "subsample": 0.8577474710637487}, #, "n_estimators": 1000
    },
    {
    "MVAtype":"XGB_combine_refit_mllgErr",
    "Label":"refit mllgErr",
    "addwei":"refit_mllgErr",
    "features":["costheta","cosTheta","Phi","mllgptdmllg","phores","dR_lg","maxdR_lg","phoCalIDMVA1","lepEta1","lepEta2","phoEta1"],
    "features_unit":["$cos(\\theta)$","$cos(\\Theta)$","$\\Phi$","$p^{T}_{H}/m_{H}$","photon resol.","min $\\Delta R(l,\\gamma)$","max $\\Delta R(l,\\gamma)$","phoID MVA","$\eta^{l1}$","$\eta^{l2}$","$\eta^{\gamma}$"],
    "feature_bins":[30,30,30,np.linspace(0, 2.5, 31),np.linspace(0, 0.3, 31),30,30,30,30,30,30],
    "hyperopt":True,
    "ModelParams":{"objective":"binary:logistic","eval_metric":"auc","tree_method":"gpu_hist","random_state":RandomState},
    "HyperParams":{"learning_rate": 0.2, "max_delta_step": 4,"max_depth": 7,"min_child_weight": 630.4529930216995, "min_split_loss": 1.619430939862514, "subsample": 0.7812801711291046}, #, "n_estimators": 1000
    },
]