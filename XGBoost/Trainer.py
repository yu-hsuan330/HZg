import os, sys
import pickle
import time
import warnings
import numpy as np
import hyperopt as hpt
from hyperopt import hp
from hyperopt.early_stop import no_progress_loss
if not sys.warnoptions:
    warnings.filterwarnings("ignore")
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

from Tools.PlotsUtils import *
from Tools.dfUtils import df_from_rootfiles, df_balance_rwt

space = {
    # "n_estimators": 150,
    "learning_rate": hp.quniform('learning_rate', 0.01, 0.1, 0.005),
    "max_depth":  hp.choice("max_depth", np.arange(2, 7)), # A problem with max_depth casted to float instead of int with the hp.quniform method.
    "max_delta_step": hp.choice("max_delta_step", np.arange(1, 20)),
    "min_child_weight": hp.uniform("min_child_weight", 1., 20.), # hp.quniform("min_child_weight", 1, 20, 1),
    "subsample": hp.uniform("subsample", 0.5, 1.),
    "min_split_loss": hp.uniform("min_split_loss", 0.1, 6), # a.k.a. gamma
}

def objective(params):
    clf = XGBClassifier(use_label_encoder=False, early_stopping_rounds=20, **params)
    clf.fit(x_train, y_train, sample_weight=w_train, verbose=0, eval_set=[(x_train, y_train), (x_test, y_test)], sample_weight_eval_set=[w_train, w_test])
    
    result = clf.evals_result()
    score = max(result["validation_1"]["auc"])
    return -score

def PrepDataset(df, TrainIndices, TestIndices, features, weight):
    X_train = df.loc[TrainIndices, features]
    Y_train = df.loc[TrainIndices, "Category"]
    W_train = df.loc[TrainIndices, weight]

    X_test = df.loc[TestIndices, features]
    Y_test = df.loc[TestIndices, "Category"]
    W_test = df.loc[TestIndices, weight]
    return np.asarray(X_train), np.asarray(Y_train), np.asarray(W_train), np.asarray(X_test), np.asarray(Y_test), np.asarray(W_test)

if __name__ == "__main__":
    
    # get the information from config file
    TrainConfig = sys.argv[1]
    importConfig=TrainConfig.replace("/", ".")
    exec("import "+importConfig+" as Conf")
    # sampleConfig = sys.argv[2]
    # importConfig2=sampleConfig.replace("/", ".")
    # exec("import "+importConfig2+" as sampleConf")   

    if Conf.Debug == True:
        prGreen("Running in debug mode : Only every 10th event will be used")

    # make directory
    if os.path.exists(Conf.OutputDirName) == 0:
        os.system("mkdir -p " + Conf.OutputDirName+"/{CodeANDConfig,Plots,Minitrees}")
    
    if len(Conf.MVAs) > 0:
        for MVAd in Conf.MVAs:
            if os.path.exists(Conf.OutputDirName+"/"+MVAd["MVAtype"]) == 0:
                prGreen("Making output directory: "+MVAd["MVAtype"])
                os.system("mkdir -p " + Conf.OutputDirName+"/"+MVAd["MVAtype"])
    
    # copy the training code
    os.system("cp "+TrainConfig+".py ./"+ Conf.OutputDirName+"/CodeANDConfig/")
    os.system("cp Trainer.py ./"+ Conf.OutputDirName+"/CodeANDConfig/")

    # load root file
    df = df_from_rootfiles(Conf.processes, Conf.Tree, Conf.branches, "False", Conf.Debug)

    # Category(num) <-> Class(string)
    df["Category"] = 0
    for i, k in enumerate(Conf.Classes):
        df.loc[df.Class == k, "Category"] = i

    # separate the train and test dataset
    index = df.index
    TrainIndices, TestIndices = [], []

    for myclass in Conf.Classes:
        condition = df["Class"] == myclass
        Indices = index[condition].values.tolist()
        myclassTrainIndices, myclassTestIndices = train_test_split(Indices, test_size=Conf.testsize, random_state=Conf.RandomState, shuffle=True) 
        TrainIndices = TrainIndices + myclassTrainIndices
        TestIndices  = TestIndices  + myclassTestIndices

    df.loc[TrainIndices, "Dataset"] = "Train"
    df.loc[TestIndices, "Dataset"] = "Test"

    df.loc[TrainIndices, "TrainDataset"] = 1
    df.loc[TestIndices, "TrainDataset"] = 0
    
    # start training !
    for MVA in Conf.MVAs:
        start_time = time.time()
        prGreen("Start" + MVA["MVAtype"] + ":")

        # apply the mass resolution as additional weight (addwei) on the signal samples (Category == 0)
        # nowei: without additional weight; varName: (1 / specific variable) as additional weight
        df["add_weight"] = df["weight"].abs()
        if MVA["addwei"] != "nowei":
            df.loc[df["Category"] == 0, "add_weight"] /= df.loc[df["Category"] == 0, MVA["addwei"]]
        
        # balance two classes
        weight = "balancedWt" # the final weight used in machine learing
        if Conf.Debug == True: print( "Balanced reweighting for training sample", flush=True)
        df.loc[TrainIndices, weight] = df_balance_rwt(df.loc[TrainIndices], SumWeightCol="add_weight", NewWeightCol=weight, Classes=Conf.Classes, debug=Conf.Debug)
        if Conf.Debug == True: print( "Balanced reweighting for testing sample" , flush=True)
        df.loc[TestIndices, weight] = df_balance_rwt(df.loc[TestIndices], SumWeightCol="add_weight", NewWeightCol=weight, Classes=Conf.Classes, debug=Conf.Debug)
        
        # prepare the training/testing dataset
        global x_train, y_train, w_train, x_test, y_test, w_test 
        x_train, y_train, w_train, x_test, y_test, w_test = PrepDataset(df, TrainIndices, TestIndices, MVA["features"], weight) 
        
        # search hyperparameter  
        best_params = {**MVA["ModelParams"], **MVA["HyperParams"]}
        if MVA["hyperopt"]:
            search_params = {**MVA["ModelParams"], **space}
            trials = hpt.Trials()
            best = hpt.fmin(fn=objective, space=search_params, algo=hpt.tpe.suggest, max_evals=500, 
                            early_stop_fn=no_progress_loss(iteration_stop_count=30, percent_increase=0.001), show_progressbar=True)
            best_params = hpt.space_eval(search_params, best)
            print("Optimized hyper parameters", best_params)
        
        # train the Classifier
        clf = XGBClassifier(use_label_encoder=False, early_stopping_rounds=20, **best_params)
        clf.fit(x_train, y_train, sample_weight=w_train, verbose=0, eval_set=[(x_train, y_train), (x_test, y_test)], sample_weight_eval_set=[w_train, w_test])

        # save the model
        clf.save_model("{}/{}/{}_best_modelXGB.txt".format(Conf.OutputDirName, MVA["MVAtype"], MVA["MVAtype"]))
        
        # plot the evaluation
        result = clf.evals_result()
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(result["validation_0"]["auc"], label="Training", linewidth=4, color="#22577E")
        ax.plot(result["validation_1"]["auc"], label="Testing", linewidth=4, color="#EF5B5B")
        ax.legend(title=MVA["Label"], loc="best", title_fontsize=14, fontsize=14, frameon=False)
        fig.savefig("{}/{}/AUC_eval_{}.pdf".format(Conf.OutputDirName, MVA["MVAtype"], MVA["MVAtype"]), bbox_inches='tight')

        # predict BDT score
        y_train_pred = clf.predict_proba(x_train)
        y_test_pred  = clf.predict_proba(x_test)

        plot_Features(MVA, df, "add_weight", Conf.OutputDirName)
        plot_VarImportance(MVA, clf, Conf)
        plot_ROC(MVA, y_train, y_test, y_train_pred, y_test_pred, w_train, w_test, Conf)
        plot_MVA(MVA, y_train, y_test, y_train_pred, y_test_pred, w_train, w_test, Conf, False)   
        
        # MVAboundary = plot_BDT(MVA, clf, Conf, sampleConf) # MVAboundary = plot_BDT(MVA, cv, Conf, sampleConf)
        # save_training(MVA["MVAtype"], clf, Conf, sampleConf, MVAboundary)        
        
        seconds = time.time() - start_time
        print("[INFO] Time Taken: {}".format(time.strftime("%H:%M:%S",time.gmtime(seconds))))
