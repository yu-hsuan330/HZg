import ROOT

# Function to extract the sigma effective of a histogram
def getEffSigma(var, weight):
    _h = ROOT.TH1D("myHist", "", 130, 105, 170)
    for i, vi in enumerate(var):
        _h.Fill(vi, weight[i])
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
                print " --> Reach nBins in effSigma calc: %s. Returning 0 for effSigma"%_h.GetName()
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
                print " --> Reach 0 in effSigma calc: %s. Returning 0 for effSigma"%_h.GetName()
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
    return wmin
    
# calculate the yield in each bin in mass window (125GeV +/- 2 sigma)
def getyield(df, sigbins=20):
    yield_ = []
    for i in range(sigbins):
        # specific signal efficiency
        low_bound, high_bound = i/sigbins, (i+1)/sigbins
        # specific mass window
        EffSigma = getEffSigma(df["refit_mllg"], df["Weight"])
        low_mass, high_mass = (125.-2*EffSigma), (125.-2*EffSigma)
        # define condition
        inRegion = (df["TBDT"]>low_bound) & (df["TBDT"]<=high_bound)
        inMassWindow = (df["refit_mllg"]>=low_mass) & (df["refit_mllg"]<=high_mass)
        yield_.append(df.loc[inRegion & inMassWindow, "Weight"].sum())
    
    return np.asarray(yield_)

def optimizeCate(sig, bkg): # set 20 bins
    #optimization
    CombSign = 0
    CombSignBou, MVAboundary = [], []
    detail = []
    for i in range(2, len(sig)+1): # choose 2-10 categories
        tCombSignMax, tCombSignMax_Last = 0, 0
        tCombSignMax_each = []
        tCombSignMax_boundaries = []
        for boundaries in combinations(range(len(sig)-1),i-1): 
        # change the boundaries on different signal efficiency
            tCombSign = 0
            tCombSign_Last = 0
            tCombSign_each = []
            for j, boundary in enumerate(boundaries):
                if j == 0 :
                    # tCombSign += sum(sig[0:boundary+1])**2/(sum(bkg[0:boundary+1]))
                    tCombSign_Last = sum(sig[0:boundary+1])**2/(sum(bkg[0:boundary+1]))
                    tCombSign_each += [sum(sig[0:boundary+1]), (sum(bkg[0:boundary+1]))]
                    # print('first',tCombSign, sum(sig[0:boundary+1]))
                    if boundary == boundaries[-1] :
                        tCombSign += sum(sig[boundaries[-1]+1:len(sig)])**2/(sum(bkg[boundaries[-1]+1:len(sig)]))
                        tCombSign_each += [sum(sig[boundaries[-1]+1:len(sig)]), (sum(bkg[boundaries[-1]+1:len(sig)]))]
                        # print('final',tCombSign, sum(sig[boundaries[-1]+1:len(sig)]))
                else :
                    tCombSign += sum(sig[boundaries[j-1]+1:boundary+1])**2/(sum(bkg[boundaries[j-1]+1:boundary+1]))
                    tCombSign_each += [sum(sig[boundaries[j-1]+1:boundary+1]),(sum(bkg[boundaries[j-1]+1:boundary+1]))]
                    # print('mid',tCombSign, sum(sig[boundaries[j-1]+1:boundary+1]))
                    if boundary == boundaries[-1] :
                        tCombSign += sum(sig[boundaries[-1]+1:len(sig)])**2/(sum(bkg[boundaries[-1]+1:len(sig)]))
                        tCombSign_each += [sum(sig[boundaries[-1]+1:len(sig)]), (sum(bkg[boundaries[-1]+1:len(sig)]))]
                        # print('final',tCombSign, sum(sig[boundaries[-1]+1:len(sig)]))
            if tCombSign > tCombSignMax :
            # the best result in the specific n categories
                tCombSignMax = tCombSign
                tCombSignMax_Last = tCombSign_Last                    
                tCombSignMax_boundaries = np.asarray(boundaries)
                tCombSignMax_each = tCombSign_each
        # print(tCombSignMax_boundaries, fn_(tCombSignMax_boundaries*0.1+0.1))
        # print(tCombSignMax-tCombSignMax_Last, tCombSignMax_Last, tCombSignMax_each)
        # print("---")
        if (tCombSignMax-CombSign)/CombSign > 0.01:
            CombSign = tCombSignMax
            CombSignBou = tCombSignMax_boundaries
            MVAboundary = fn_(CombSignBou/len(sig)+1./len(sig))
            # MVAboundary = fn_(CombSignBou*0.1+0.1)
            detail = tCombSignMax_each

        else:
            print("choose the results:", CombSign)
            print(CombSignBou, MVAboundary)
            print(detail)
            break
    return MVAboundary

'''
def categorization(sig, bkg):
            CombSign = 0
            CombSignBou, MVAboundary = [], []
            detail = []
            for i in range(2, 21): # choose 2-10 categories
                tCombSignMax, tCombSignMax_Last = 0, 0
                tCombSignMax_each = []
                tCombSignMax_boundaries = []
                for boundaries in combinations(range(len(sig)-1),i-1): 
                # change the boundaries on different signal efficiency
                    tCombSign = 0
                    tCombSign_Last = 0
                    tCombSign_each = []
                    for j, boundary in enumerate(boundaries):
                        if j == 0 :
                            # tCombSign += sum(sig[0:boundary+1])**2/(sum(bkg[0:boundary+1]))
                            tCombSign_Last = sum(sig[0:boundary+1])**2/(sum(bkg[0:boundary+1]))
                            tCombSign_each += [sum(sig[0:boundary+1]), (sum(bkg[0:boundary+1]))]
                            # print('first',tCombSign, sum(sig[0:boundary+1]))
                            if boundary == boundaries[-1] :
                                tCombSign += sum(sig[boundaries[-1]+1:len(sig)])**2/(sum(bkg[boundaries[-1]+1:len(sig)]))
                                tCombSign_each += [sum(sig[boundaries[-1]+1:len(sig)]), (sum(bkg[boundaries[-1]+1:len(sig)]))]
                                # print('final',tCombSign, sum(sig[boundaries[-1]+1:len(sig)]))
                        else :
                            tCombSign += sum(sig[boundaries[j-1]+1:boundary+1])**2/(sum(bkg[boundaries[j-1]+1:boundary+1]))
                            tCombSign_each += [sum(sig[boundaries[j-1]+1:boundary+1]),(sum(bkg[boundaries[j-1]+1:boundary+1]))]
                            # print('mid',tCombSign, sum(sig[boundaries[j-1]+1:boundary+1]))
                            if boundary == boundaries[-1] :
                                tCombSign += sum(sig[boundaries[-1]+1:len(sig)])**2/(sum(bkg[boundaries[-1]+1:len(sig)]))
                                tCombSign_each += [sum(sig[boundaries[-1]+1:len(sig)]), (sum(bkg[boundaries[-1]+1:len(sig)]))]
                                # print('final',tCombSign, sum(sig[boundaries[-1]+1:len(sig)]))
                    if tCombSign > tCombSignMax :
                    # the best result in the specific n categories
                        tCombSignMax = tCombSign
                        tCombSignMax_Last = tCombSign_Last
                        tCombSignMax_boundaries = np.asarray(boundaries)
                        tCombSignMax_each = tCombSign_each
                # print(tCombSignMax_boundaries, fn_(tCombSignMax_boundaries*0.1+0.1))
                # print(tCombSignMax-tCombSignMax_Last, tCombSignMax_Last, tCombSignMax_each)
                # print("---")
                
                if (tCombSignMax-CombSign)/CombSign > 0.01:
                    CombSign = tCombSignMax
                    CombSignBou = tCombSignMax_boundaries
                    MVAboundary = fn_(CombSignBou*0.05+0.05)
                    # MVAboundary = fn_(CombSignBou*0.1+0.1)
                    detail = tCombSignMax_each

                else:
                    print("choose the results:", CombSign)
                    print(CombSignBou, MVAboundary)
                    print(detail)
                    break
            return MVAboundary
'''

'''
def optimizeCate(sig, bkg, fn_):
    CombSign = 0
    CombSignBou, MVAboundary = [], []
    detail = []
    for i in range(2, len(sig)+1): # choose 2-10 categories
        tCombSignMax, tCombSignMax_Last = 0, 0
        tCombSignMax_each = []
        tCombSignMax_boundaries = []
        for boundaries in combinations(range(len(sig)-1),i-1): 
        # change the boundaries on different signal efficiency
            tCombSign = 0
            tCombSign_Last = 0
            tCombSign_each = []
            for j, boundary in enumerate(boundaries):
                if j == 0 :
                    # tCombSign += sum(sig[0:boundary+1])**2/(sum(bkg[0:boundary+1]))
                    tCombSign_Last = sum(sig[0:boundary+1])**2/(sum(bkg[0:boundary+1]))
                    tCombSign_each += [sum(sig[0:boundary+1]), (sum(bkg[0:boundary+1]))]
                    # print('first',tCombSign, sum(sig[0:boundary+1]))
                    if boundary == boundaries[-1] :
                        tCombSign += sum(sig[boundaries[-1]+1:len(sig)])**2/(sum(bkg[boundaries[-1]+1:len(sig)]))
                        tCombSign_each += [sum(sig[boundaries[-1]+1:len(sig)]), (sum(bkg[boundaries[-1]+1:len(sig)]))]
                        # print('final',tCombSign, sum(sig[boundaries[-1]+1:len(sig)]))
                else :
                    tCombSign += sum(sig[boundaries[j-1]+1:boundary+1])**2/(sum(bkg[boundaries[j-1]+1:boundary+1]))
                    tCombSign_each += [sum(sig[boundaries[j-1]+1:boundary+1]),(sum(bkg[boundaries[j-1]+1:boundary+1]))]
                    # print('mid',tCombSign, sum(sig[boundaries[j-1]+1:boundary+1]))
                    if boundary == boundaries[-1] :
                        tCombSign += sum(sig[boundaries[-1]+1:len(sig)])**2/(sum(bkg[boundaries[-1]+1:len(sig)]))
                        tCombSign_each += [sum(sig[boundaries[-1]+1:len(sig)]), (sum(bkg[boundaries[-1]+1:len(sig)]))]
                        # print('final',tCombSign, sum(sig[boundaries[-1]+1:len(sig)]))
            if tCombSign > tCombSignMax :
            # the best result in the specific n categories
                tCombSignMax = tCombSign
                tCombSignMax_Last = tCombSign_Last                    
                tCombSignMax_boundaries = np.asarray(boundaries)
                tCombSignMax_each = tCombSign_each
        # print(tCombSignMax_boundaries, fn_(tCombSignMax_boundaries*0.1+0.1))
        # print(tCombSignMax-tCombSignMax_Last, tCombSignMax_Last, tCombSignMax_each)
        # print("---")
        if (tCombSignMax-CombSign)/CombSign > 0.01:
            CombSign = tCombSignMax
            CombSignBou = tCombSignMax_boundaries
            MVAboundary = fn_(CombSignBou/len(sig)+1./len(sig))
            # MVAboundary = fn_(CombSignBou*0.1+0.1)
            detail = tCombSignMax_each

        else:
            print("choose the results:", CombSign)
            print(CombSignBou, MVAboundary)
            print(detail)
            break
    return MVAboundary
'''