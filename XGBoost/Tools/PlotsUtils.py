# import ROOT
import numpy as np
import pandas as pd
# import dask.dataframe as dd
import math
from xgboost import plot_importance
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from scipy.interpolate import interp1d
from sklearn import metrics, preprocessing
from itertools import combinations
# from CateUtils import *
from .dfUtils import *

def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))

def ToCategorical(y, num_classes = None, dtype = "float64"):
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def pltSty(ax, xName = "x-axis", yName = "y-axis", TitleSize = 17, LabelSize = 16, TickSize = 13, MajTickLength = 7, MinTickLength = 4, yAuto = True):
    ax.set_xlabel(xName, fontsize = LabelSize, loc = "right")
    ax.set_ylabel(yName, fontsize = LabelSize, loc = "top")
    # ax.text(1, 1, "(13 TeV)", horizontalalignment = "right", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize)
    ax.text(0, 1.01, "CMS", horizontalalignment = "left", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize * 1.3, fontweight = "bold")
    ax.text(TitleSize * 0.01, 1.015, "work-in-progress", horizontalalignment = "left", verticalalignment = "bottom", transform = ax.transAxes, fontsize = TitleSize)

    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if yAuto :
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction = "in", length = MajTickLength, labelsize = TickSize, top = True, right = True)
    ax.tick_params(direction = "in", length = MinTickLength, which = "minor", labelsize = TickSize, top = True, right = True)

def plot_Features(MVA, df, weight, OutputDirName):
    # input the feature units and the weight
    features_unit, feature_bins = MVA["features_unit"], MVA["feature_bins"]
    nFeature = len(features_unit)+1 # due to photon ID MVA->EE/EB
    wsig = np.asarray(df[weight][df['Category']==0])
    wbkg = np.asarray(df[weight][df['Category']==1])
    
    row = math.ceil(nFeature/4)
    fig, axes = plt.subplots(row, 4, figsize=(7*4, 6*(row)))
    i_, add = 0, 0
    for i, feature in enumerate(MVA["features"]):
        # input the feature in the signal/background
        sig = np.asarray(df[feature][df['Category']==0])
        bkg = np.asarray(df[feature][df['Category']==1])
        unit = features_unit[i]
        i_ = i + add
        if feature == "phoCalIDMVA1":
            df['phoEE_EB'] = 0
            df['phoEE_EB'][df['phoSCEta1'].abs()>1.566] = 0
            df['phoEE_EB'][df['phoSCEta1'].abs()<1.4442] = 1
            for j, region in enumerate(["(EE)","(EB)"]):
                sig_ = np.asarray(df[feature][(df['Category']==0)&(df['phoEE_EB']==j)])
                bkg_ = np.asarray(df[feature][(df['Category']==1)&(df['phoEE_EB']==j)])
                wsig_ = np.asarray(df[weight][(df['Category']==0)&(df['phoEE_EB']==j)])
                wbkg_ = np.asarray(df[weight][(df['Category']==1)&(df['phoEE_EB']==j)])
                ax = axes[(i_+j)//4, (i_+j)-((i_+j)//4)*4]
                ax.hist(sig_, label="Signal", weights=wsig_, bins=feature_bins[i], density=True,
                    histtype='stepfilled',alpha=0.4, linewidth=2, color='#22577E')
                ax.hist(bkg_, label="Background", weights=wbkg_, bins=feature_bins[i], density=True,
                    histtype='stepfilled',alpha=0.4, linewidth=2, color='#EF5B5B')
                pltSty(ax, xName=features_unit[i]+region, yName="Events", yAuto=False)
                ax.legend(title=MVA["Label"], loc="best", title_fontsize=12, fontsize=12, frameon=False)
            add = add+1
            continue
        ax = axes[(i_)//4, i_-(i_//4)*4]
        
        if feature == "jetEta1":
            sig[sig==0], bkg[bkg==0] = -99, -99
        
        ax.hist(sig, label="Signal", weights=wsig, bins=feature_bins[i], density=True,
                histtype='stepfilled', alpha=0.4, linewidth=2, color='#22577E')
        ax.hist(bkg, label="Background", weights=wbkg, bins=feature_bins[i], density=True,
                histtype='stepfilled', alpha=0.4, linewidth=2, color='#EF5B5B')
            
        pltSty(ax, xName=features_unit[i], yName="Events", yAuto=False)
        ax.legend(title=MVA["Label"], loc="best", title_fontsize=12, fontsize=12, frameon=False)

    fig.savefig(OutputDirName+"/"+MVA["MVAtype"]+"/"+"Feature_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    fig.savefig(OutputDirName+"/Plots/"+"Feature_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')

def plot_VarImportance(MVA, model, Conf):
    bst = model.get_booster()
    bst.feature_names = MVA["features"]
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    plot_importance(bst, ax = ax, importance_type = "gain", height = 0.7, grid = False, title = None,
         xlabel = None, ylabel = None, show_values=0, color = "#A2D5AB")
    
    pltSty(ax, xName = "Importance", yName = "Features", yAuto = False)
    plt.draw()
    fig.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"VI_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    fig.savefig(Conf.OutputDirName+"/Plots/"+"VI_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    plt.close("all")

def plot_ROC(MVA, y_train, y_test, y_train_pred, y_test_pred, w_train, w_test, Conf):
    y_train_categorical = ToCategorical(y_train, num_classes=2)
    y_test_categorical = ToCategorical(y_test, num_classes=2)

    fpr, tpr, th = metrics.roc_curve(y_test_categorical[:,0], y_test_pred[:,0], sample_weight=w_test)
    fpr_tr, tpr_tr, th_tr = metrics.roc_curve(y_train_categorical[:,0], y_train_pred[:,0], sample_weight=w_train)
    roc_auc = metrics.auc(fpr, tpr)
    roc_auc_tr = metrics.auc(fpr_tr, tpr_tr)
    bkgrej = (1 - fpr)
    bkgrej_tr = (1 - fpr_tr)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    pltSty(ax, xName = "Signal efficiency", yName = "Background rejection", yAuto = False)
    ax.plot(tpr_tr, bkgrej_tr, label='Training  AUC=%2.1f%%' % (roc_auc_tr*100), linewidth=4, color="#22577E")
    ax.plot(tpr, bkgrej, label='Testing   AUC=%2.1f%%' % (roc_auc*100), linewidth=4, color="#EF5B5B", linestyle="dashed")
    ax.legend(title=MVA["Label"], loc="lower left", title_fontsize=15, fontsize=15, frameon=False)
    fig.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"ROC_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    fig.savefig(Conf.OutputDirName+"/Plots/"+"ROC_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')

def plot_MVA(MVA, y_train, y_test, y_train_pred, y_test_pred, w_train, w_test, Conf, transform=False):
    n_classes = len(Conf.Classes)
    y_train_categorical = ToCategorical(y_train, n_classes)
    y_test_categorical = ToCategorical(y_test,n_classes)

    figMVA, axMVA = plt.subplots(1, 1, figsize=(6, 6))
    xmin = 0
    if transform:
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), clip=True)
        min_max_scaler.fit(y_train_pred)
        y_train_pred = min_max_scaler.transform(y_train_pred)
        y_test_pred = min_max_scaler.transform(y_test_pred)
        xmin = -1
    for k in range(n_classes):
        axMVA.hist(y_test_pred[:, 0][y_test_categorical[:, k]==1], bins=np.linspace(xmin, 1, 41), label=Conf.Classes[k]+'_test',
                    weights=w_test[y_test_categorical[:, k]==1]/np.sum(w_test[y_test_categorical[:, k]==1]),
                    histtype='step',linewidth=2,color=Conf.ClassColors[k])
        axMVA.hist(y_train_pred[:, 0][y_train_categorical[:, k]==1],bins=np.linspace(xmin, 1, 41),label=Conf.Classes[k]+'_train',
                    weights=w_train[y_train_categorical[:, k]==1]/np.sum(w_train[y_train_categorical[:, k]==1]),
                    histtype='stepfilled',alpha=0.3,linewidth=2,color=Conf.ClassColors[k])
    pltSty(axMVA, xName="Score", yName="Events", yAuto=False)
    # axMVA.set_ylim([0,0.09])
    if Conf.channel == "ele":
        axMVA.legend(title=MVA["Label"], loc="upper center", title_fontsize=12, fontsize=12, frameon=False)
    else:    
        axMVA.legend(title=MVA["Label"], loc="best", title_fontsize=12, fontsize=12, frameon=False)
    if Conf.MVAlogplot:
        axMVA.set_yscale('log')

    if transform :
        figMVA.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"MVAt_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
        figMVA.savefig(Conf.OutputDirName+"/Plots/"+"MVAt_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    else:
        figMVA.savefig(Conf.OutputDirName+"/"+MVA["MVAtype"]+"/"+"MVA_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
        figMVA.savefig(Conf.OutputDirName+"/Plots/"+"MVA_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')


# calculate the yield in each bin in mass window (mean +/- 2 sigma)
def getyield(df, low_bound, high_bound, _mean=0, _effsig=0):
    # specific signal efficiency
    inRegion = (df["TBDT"]>low_bound) & (df["TBDT"]<=high_bound)
    # specific mass window
    if (_mean==0) & (_effsig==0):
        # mean, EffSigma = getEffSigma(df.loc[inRegion, "refit_mllg"], df.loc[inRegion, "Weight"])
        mean, EffSigma = 125, 2.5
    else:
        mean, EffSigma = _mean, _effsig
    low_mass, high_mass = (mean-2*EffSigma), (mean+2*EffSigma)
    # low_mass, high_mass = 105, 170
    inMassWindow = (df["refit_mllg"]>=low_mass) & (df["refit_mllg"]<=high_mass)
    yield_ = df.loc[inRegion & inMassWindow, "Weight"].sum()
    
    return yield_, mean, EffSigma

# Function to extract the sigma effective of a histogram
def getEffSigma(var, weight):
    _h = ROOT.TH1D("myHist", "", 130, 105, 170)
    for vi, wi in zip(var, weight):
        _h.Fill(vi, wi)
    nbins, binw, xmin = _h.GetXaxis().GetNbins(), _h.GetXaxis().GetBinWidth(1), _h.GetXaxis().GetXmin()
    mu, rms, total = _h.GetMean(), _h.GetRMS(), _h.Integral()
    # Scan round window of mean: window RMS/binWidth (cannot be bigger than 0.1*number of bins)
    nWindow = int(rms/binw) if (rms/binw) < 0.1*nbins else int(0.1*nbins)
    # Determine minimum width of distribution which holds 0.693 of total
    rlim = 0.683*total
    wmin, iscanmin = 9999999, -999
    for iscan in range(-1*nWindow,nWindow+1):
        # Find bin idx in scan: iscan from mean
        i_centre = int((mu-xmin)/binw+1+iscan)
        x_centre = (i_centre-0.5)*binw+xmin # * 0.5 for bin centre
        x_up, x_down = x_centre, x_centre
        i_up, i_down = i_centre, i_centre
        # Define counter for yield in bins: stop when counter > rlim
        y = _h.GetBinContent(i_centre) # Central bin height
        r = y
        reachedLimit = False
        for j in range(1,nbins):
            if reachedLimit: continue
            # Up:
            if(i_up < nbins)&(not reachedLimit):
                i_up+=1
                x_up+=binw
                y = _h.GetBinContent(i_up) # Current bin height
                r+=y
                if r>rlim: reachedLimit = True
            else:
                print(" --> Reach nBins in effSigma calc: {}. Returning 0 for effSigma".format(_h.GetName()))
                return 0
            # Down:
            if( not reachedLimit ):
                if(i_down > 0):
                    i_down-=1
                    x_down-=binw
                    y = _h.GetBinContent(i_down) #Current bin height
                    r+=y
                    if r>rlim: reachedLimit = True
                else:
                    print(" --> Reach 0 in effSigma calc: {}. Returning 0 for effSigma".format(_h.GetName())) 
                    return 0
        # Calculate fractional width in bin takes above limt (assume linear)
        if y == 0.: dx = 0.
        else: dx = (r-rlim)*(binw/y)
        # Total width: half of peak
        w = (x_up-x_down+binw-dx)*0.5
        if w < wmin:
            wmin = w
            iscanmin = iscan
    # Return effSigma
    return mu, wmin

def optimizeCate(sig, bkg, fn_):
    def get_yield(df, lbound, hbound):
        # specific mass window & signal efficiency
        low_mass, high_mass = 120, 130
        inMassWindow = (df["refit_mllg"]>=low_mass) & (df["refit_mllg"]<=high_mass)
        inRegion = (df["TBDT"]>lbound) & (df["TBDT"]<=hbound)
        return df.loc[inRegion & inMassWindow, "Weight"].sum()

    def get_signif(sig, bkg, boundary):
        combined_signif, low_bound, high_bound = 0., 0., 0.
        for i in np.sort(boundary):
            low_bound = high_bound
            high_bound = i
            # calculate the yield/combined significance
            ysig, ybkg = get_yield(sig, low_bound, high_bound), get_yield(bkg, low_bound, high_bound)
            combined_signif += ysig**2/ybkg
        return combined_signif

    Boundaries = np.array([0.3])
    imporve, signif, signif_ = 99., 0.001, 0.001
    f = open("./optimization.txt", 'w')

    while imporve > 0.01:
        signif = signif_
        Boundaries = np.append(Boundaries, -1.)
        tsignif = np.array([])
        for i in range(1, 3):
            if 0.1*i in Boundaries[:-1]: 
                tsignif = np.append(tsignif, 0)
                continue
            Boundaries[-1] = 0.1*i
            tsignif = np.append(tsignif, get_signif(sig, bkg, np.append(Boundaries, 1.)))
        print(tsignif, file=f)
        signif_, Boundaries[-1] = tsignif.max(), 0.1*(1+tsignif.argmax())
        imporve = (signif_-signif)/signif
    print(signif_)
    print(Boundaries[:-1])
    f.close()
    return fn_(np.sort(Boundaries[:-1]))

def plot_BDT(MVA, model, model2, Conf, sampleConf, Norm=False, Transform=True):
    fig, ax = plt.subplots(1, 1, figsize=(6.5,6))
    pltSty(ax, xName = "BDT score", yName = "Events", yAuto=False)
    pred_stack, _pred_stack, weight_stack, label_stack, color_stack = [], [], [], [], []
    # df_signal, df_bkg = pd.DataFrame(), pd.DataFrame()
    df_signal, df_bkg = [], []
    for sample in sampleConf.samples:
        if sample["Production"] == "Data": continue
        df = Predict(Conf.channel, sample, model, sampleConf, Conf)
        pred, weight = np.asarray(df["kinMVA"]), np.asarray(df["Weight"])
        if sample["Production"] == "Signal":
            # df_signal = df_signal.append(df, ignore_index=True)
            df_signal.append(df)
            sig_count, sig_bins_count = np.histogram(pred, bins=np.linspace(0, 1, 2001), weights=weight)
            ax.hist(pred, bins=np.linspace(0, 1, 51), label=sample["Label"]+"x100", weights=weight*100, 
                        histtype='step', stacked=False, fill=False, linewidth=3, color=sample["Color"], density=Norm)
        else:
            # df_bkg = df_bkg.append(df, ignore_index=True)
            df_bkg.append(df)
            pred_stack.append(pred)
            weight_stack.append(weight)
            label_stack.append(sample["Label"])
            color_stack.append(sample["Color"])
    df_signal = pd.DataFrame(df_signal)
    df_bkg = pd.DataFrame(df_bkg)

    ax.hist(pred_stack, bins=np.linspace(0, 1, 51), weights=weight_stack,label=label_stack, color=color_stack,
                    histtype='bar', stacked=True, linewidth=2, density=Norm)
    ax.legend(title=MVA["Label"], loc="best", title_fontsize=14, fontsize=14, frameon=False)
    ax.set_yscale('log')
    fig.savefig("{}/{}/BDT_{}.pdf".format(Conf.OutputDirName, MVA["MVAtype"], MVA["MVAtype"]), bbox_inches='tight')
    fig.savefig("{}/Plots/BDT_{}.pdf".format(Conf.OutputDirName, MVA["MVAtype"]), bbox_inches='tight')

    if Transform:
        # get cumulative distribution and intepolation function
        cdf = np.cumsum(sig_count) / sum(sig_count)
        # fn = interp1d(np.linspace(0, 1, num=2000), cdf)
        # fn_ = interp1d(cdf, np.linspace(0, 1, num=2000))
        fn = interp1d(sig_bins_count, np.append([0.], cdf))
        fn_ = interp1d(np.append([0.], cdf), sig_bins_count)
        figT, axT = plt.subplots(1, 1, figsize=(6.5,6))
        pltSty(axT, xName = "Transformed BDT score", yName = "Events", yAuto=False)
        
        # transformed BDT - signal sample
        _pred, weight = fn(np.asarray(df_signal["kinMVA"])), np.asarray(df_signal["Weight"])
        axT.hist(_pred, bins=np.linspace(0, 1, 11), label=sampleConf.samples[0]["Label"], weights=weight,
                        histtype='step', stacked=False, fill=False, linewidth=3, color=sampleConf.samples[0]["Color"], density=Norm)
        # transformed BDT - bkg sample
        for i in range(len(pred_stack)):
            _pred_stack.append(fn(np.asarray(pred_stack[i]))) # _pred_stack.append() = fn(np.asarray(pred_stack))
        axT.hist(_pred_stack, bins=np.linspace(0, 1, 11), weights=weight_stack, label=label_stack, color=color_stack,
                        histtype='bar', stacked=True, linewidth=2, density=Norm)
        # plot setting
        axT.legend(title=MVA["Label"], loc="best", title_fontsize=14, fontsize=14, frameon=False, ncol=2)
        axT.set_yscale('log')
        axT.set_ylim([0, 5*axT.get_ylim()[1]]) # axT.set_ylim([0, 50000])

        # plt.ylabel('signal-background ratio', fontsize=14) 
        figT.savefig("{}/{}/TBDT_{}.pdf".format(Conf.OutputDirName, MVA["MVAtype"], MVA["MVAtype"]), bbox_inches='tight')
        figT.savefig("{}/Plots/TBDT_{}.pdf".format(Conf.OutputDirName, MVA["MVAtype"]), bbox_inches='tight')

        # category optimization
        df_signal.loc[:,"TBDT"] = _pred
        df_bkg.loc[:,"TBDT"] = fn(df_bkg["kinMVA"])
        # yield_sig = getyield(df_signal) # yield_sig_, bin_sig_ = np.histogram(df_signal["TBDT"], bins=np.linspace(0, 1, 21), weights=df_signal["Weight"]) 
        # yield_bkg = getyield(df_bkg) # yield_bkg_, bin_bkg_ = np.histogram(df_bkg["TBDT"], bins=np.linspace(0, 1, 21), weights=df_bkg["Weight"]) 
        # print(yield_sig, yield_bkg)

        MVAboundary = optimizeCate(df_signal, df_bkg, fn_)
        return MVAboundary, fn



def plot_confusion_matrix(MVA, y, y_pred, Classes, OutputDirName, Norm=True, sample="Testing"):
    
    y = ToCategorical(y, len(Classes))

    cm = metrics.confusion_matrix(y.argmax(axis=1), y_pred.argmax(axis=1))
    if Norm:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_df = pd.DataFrame(cm, index = Classes, columns = Classes)

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    pltSty(ax, xName="Predicted class", yName="True class", yAuto=False)

    im = plt.imshow(cm_df, cmap="YlGnBu")
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize = 10)
    
    tick_marks = np.arange(len(cm_df.columns))
    plt.xticks(tick_marks, Classes, fontsize = 13)
    plt.yticks(tick_marks, Classes, fontsize = 13)

    
    fmt = ".3f" if Norm else "d"
    thresh = cm.max() / 2.
    for i in range(len(cm_df.columns)):
        for j in range(len(cm_df.index)):
            plt.text(j, i, format(cm_df.iloc[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    # ax.set_title("confusion matrix")
    plt.draw()
    fig.savefig(OutputDirName+"/"+MVA["MVAtype"]+"/"+"CM_"+MVA["MVAtype"]+"_"+sample+".pdf", bbox_inches='tight')
    plt.close("all")

def plot_correlation_matrix(MVA, train, OutputDirName):
    correlations = train.corr()    
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    pltSty(ax, xName = "Importance", yName = "Features", yAuto = False)

    tick_marks = np.arange(len(train.columns))
    plt.xticks(tick_marks, MVA["features"], fontsize = 13, rotation=90)
    plt.yticks(tick_marks, MVA["features"], fontsize = 13, rotation=90)

    im = plt.imshow(correlations)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize = 10)

    fmt = ".2f"
    thresh = correlations.max() / 2.
    for i in range(len(train.columns)):
        for j in range(len(train.index)):
            plt.text(j, i, format(train.iloc[i, j], fmt), ha="center", va="center", color="white" if train.iloc[i, j] > thresh else "black")

    plt.draw()
    fig.savefig(OutputDirName+"/"+MVA["MVAtype"]+"/"+"VC_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    fig.savefig(OutputDirName+"/Plots/"+"VC_"+MVA["MVAtype"]+".pdf", bbox_inches='tight')
    plt.close("all")

#TODO print(metrics.classification_report(y_test, predicted_y))
