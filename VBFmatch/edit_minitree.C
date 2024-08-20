#include <vector>
#include <string>

using namespace std;
using namespace ROOT::VecOps;


using RNode = ROOT::RDF::RNode;
using VecI_t = const ROOT::RVec<int>&;
using VecF_t = const ROOT::RVec<float>&; // using cRVecF = const ROOT::RVecF &;
using VecS_t = const ROOT::RVec<size_t>&;

TLorentzVector TLVec(int tag, float px, float py, float pz, float e){
    TLorentzVector v; 
    if(tag == 1) v.SetPxPyPzE(px, py, pz, e);
    return v;
};
TLorentzVector TLVec_(float pt, float eta, float phi, float e){
    TLorentzVector v; v.SetPtEtaPhiE(pt, eta, phi, e);
    return v;
};
void VBF_match(string inpath, string infile, string outpath){

    ROOT::RDataFrame df("TH", Form("%s/%s.root", inpath.c_str(), infile.c_str()));
    // auto TLVec1 = [](float px, float py, float pz, float e){
    //     TLorentzVector v; v.SetPxPyPzE(px, py, pz, e);
    //     return v;
    // };
    // auto TLVec2 = [](float pt, float eta, float phi, float e){
    //     TLorentzVector v; v.SetPtEtaPhiE(pt, eta, phi, e);
    //     return v;        
    // };
    auto df_ = df
                //  .Define("genjet1",    TLVec1,  {"lhePx1","lhePy1", "lhePz1", "lheE1"})
                 .Define("genjet1",     "TLVec(VBFtag, lhePx[1], lhePy[1], lhePz[1], lheE[1])")
                 .Define("genjet2",     "TLVec(VBFtag, lhePx[2], lhePy[2], lhePz[2], lheE[2])")
                 .Define("recojet1",    "TLVec_(jetGenPt[JetID[0]], jetGenEta[JetID[0]], jetGenPhi[JetID[0]], jetGenEn[JetID[0]])")
                 .Define("recojet2",    "TLVec_(jetGenPt[JetID[1]], jetGenEta[JetID[1]], jetGenPhi[JetID[1]], jetGenEn[JetID[1]])")
                 .Define("isVBFmatch",  "VBFtag == 1 && ((abs(genjet1.Pt()-jetGenPt[JetID[0]])/jetGenPt[JetID[0]]<0.01 && abs(genjet2.Pt()-jetGenPt[JetID[1]])/jetGenPt[JetID[1]]<0.01) || (abs(genjet1.Pt()-jetGenPt[JetID[1]])/jetGenPt[JetID[1]]<0.01 && abs(genjet2.Pt()-jetGenPt[JetID[0]])/jetGenPt[JetID[0]]<0.01))")
                 .Filter("VBFtag == 1",     "VBF tag")
                 .Filter("isVBFmatch == 1", "VBF match");

    df_.Report()->Print();
    df_.Snapshot("TH", Form("%s/%s_VBFmatch.root",outpath.c_str(), infile.c_str()));
}

void edit_minitree(){
    vector<string> files = {
        "UL16_preVFP_ele_VBF_125GeV",
        "UL16_preVFP_mu_VBF_125GeV",
        "UL16_postVFP_ele_VBF_125GeV",
        "UL16_postVFP_mu_VBF_125GeV",
        "UL17_ele_VBF_125GeV",
        "UL17_mu_VBF_125GeV",
        "UL18_ele_VBF_125GeV",
        "UL18_mu_VBF_125GeV",
    };

    for(int i=0; i<files.size(); i++){
        VBF_match("../../minitrees_test", files[i], "minitrees_HBID");
    }
    


}
 