#=============================================#
#           training configurations           #
#       combined(e+u) channel / UL dataset    #
#               Run-2 variables               #
#=============================================# 
import numpy as np

OutputDirName = "kinMVA_XGB_ULRun2_dedi_test" #All plots, models, config file will be stored here
Tree, channel = "TH", "combine"
Debug, MVAlogplot = False, False
testsize, RandomState = 0.5, 123

branches = ["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1", "lepEta1", "lepEta2", "phoEta1",
            "totwei", "totSF", "refit_mllgErr", "phoSCEta1"]
features = ["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1", "lepEta1", "lepEta2", "phoEta1"]

Classes, ClassColors = ["Signal","Background"], ["#22577E", "#EF5B5B"]
# selection = "1"
tag = "(VBFtag+Leptag == 0)"
processes = [
    {
    "Class":"Signal",
    "path":["../minitrees_dedicated/UL*_125GeV.root"],
    "selection":"(refit_mllg < 170) & (refit_mllg > 105) & (VBFtag+Leptag == 0)",
    "weight": ("totwei", "totSF"),
    },
    
    {
    "Class":"Background",
    "path":["../minitrees_dedicated/UL*_DYJetsToLL.root",
            "../minitrees_dedicated/UL*_SMZg.root"],
    "selection":"(refit_mllg < 170) & (refit_mllg > 105) & (VBFtag+Leptag == 0)", 
    "weight": ("totwei", "totSF"),
    },
]

MVAs = [
    {
    "MVAtype":"XGB_combine_nowei",
    "Label":"no-weight",    # shown in the legend of the plots
    "addwei":"nowei",   # nowei means no additional weight applied
    "features":["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1", "lepEta1", "lepEta2", "phoEta1"],
    "features_unit":["$cos(\\theta)$", "$cos(\\Theta)$", "$\\Phi$", "$p^{T}_{H}/m_{H}$", "photon resol.", "min $\\Delta R(l,\\gamma)$", "max $\\Delta R(l,\\gamma)$", "phoID MVA", "$\eta^{l1}$", "$\eta^{l2}$", "$\eta^{\gamma}$"],
    "feature_bins":[30, 30, 25, np.linspace(0, 2.5, 31), np.linspace(0, 0.3, 31), 30, 30, 30, 30, 30, 30],
    "hyperopt":True, # True: optimize the hyper parameters; False: use the following HyperParams value for training.
    "ModelParams":{"objective":"binary:logistic", "eval_metric":"auc", "tree_method":"gpu_hist", "random_state":RandomState},
    "HyperParams":{"learning_rate":0.09673311594155506, "max_delta_step":14, "max_depth":6, "min_child_weight":139.61189731050433, "min_split_loss":6.438958312821566, "subsample":0.4668532949233947},
    },
    {
    "MVAtype":"XGB_combine_refit_mllgErr",
    "Label":"refit mllgErr",
    "addwei":"refit_mllgErr", # refit_mllgErr weight is applied
    "features":["costheta", "cosTheta", "Phi", "mllgptdmllg", "phores", "dR_lg", "maxdR_lg", "phoCalIDMVA1", "lepEta1", "lepEta2", "phoEta1"],
    "features_unit":["$cos(\\theta)$", "$cos(\\Theta)$", "$\\Phi$", "$p^{T}_{H}/m_{H}$", "photon resol.", "min $\\Delta R(l,\\gamma)$", "max $\\Delta R(l,\\gamma)$", "phoID MVA", "$\eta^{l1}$", "$\eta^{l2}$", "$\eta^{\gamma}$"],
    "feature_bins":[30, 30, 30, np.linspace(0, 2.5, 31), np.linspace(0, 0.3, 31), 30, 30, 30, 30, 30, 30],
    "hyperopt":True,
    "ModelParams":{"objective":"binary:logistic", "eval_metric":"auc", "tree_method":"gpu_hist", "random_state":RandomState},
    "HyperParams":{"learning_rate":0.09733817721130829, "max_delta_step":2, "max_depth":5, "min_child_weight":13.787695290646582, "min_split_loss":0.21998326422122466, "subsample":0.6877187826497442},
    },
]