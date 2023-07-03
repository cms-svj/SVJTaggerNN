import torch
import torch.nn as nn
from torch.nn import functional as f
import torch.utils.data as udata
from torch.cuda.amp import autocast
import os
from models import DNN, DNN_GRF
from dataset import RootDataset
import matplotlib as mpl
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score, confusion_matrix
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import itertools
import copy
from GPUtil import showUtilization as gpu_usage
import seaborn as sns
import mplhep as hep
from dataset_sampler import getDataset
from Disco import distance_corr

mpl.rc("font", family="serif", size=18)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mplColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

plotFormat = "pdf"

def kstest(data_train,data_test):
    ks, pv = stats.ks_2samp(data_train,data_test)
    return pv

def collectiveKS(dataList):
    allComs = list(itertools.combinations(range(len(dataList)),2))
    ksSum = 0
    for com in allComs:
        data1 = np.array(dataList[com[0]])
        data2 = np.array(dataList[com[1]])
        newdata1 = data1[(data1>0) & (data2>0)]
        newdata2 = data2[(data1>0) & (data2>0)]
        ks, pv = stats.ks_2samp(newdata1,newdata2)
        ksSum += pv
    return ksSum/len(allComs)

def getCondition(inputFileNames,key,inpIndex):
    conditions = np.zeros(len(inpIndex),dtype=bool)
    for i in range(len(inputFileNames)):
        fileName = inputFileNames[i]
        if key in fileName:
            conditions = conditions | (inpIndex == i)
    return conditions

def getNNOutput(dataset, model, device, signalIndex=2):
    batchSize = 100
    labels = np.array([])
    targetLabels = np.array([])
    output_tags = np.array([])
    rawOutputs = np.array([])
    allOutputs = np.array([])
    inputFileIndices = np.array([])
    dcorrVars = np.array([])
    sts = np.array([])
    weight = np.array([])
    meds = np.array([])
    darks = np.array([])
    rinvs = np.array([])
    loader = udata.DataLoader(dataset=dataset, batch_size=batchSize, num_workers=0)
    for i, data in tqdm(enumerate(loader), unit="batch", total=len(loader)):
        data, dcorrVar, st, l, w, med, dark, rinv = data["data"], data["dcorrVar"], data["st"], data["label"], data["w"], data["mMed"], data["mDark"], data["rinv"]
        #l, points, features, masks, inputFileIndex, p, w, med, dark, rinv = data
        labels = np.concatenate((labels,l.numpy()))
        #inputFileIndices = np.concatenate((inputFileIndices,inputFileIndex.squeeze(1).numpy()))
        dcorrVars = np.concatenate((dcorrVars,dcorrVar.float().numpy()))
        sts = np.concatenate((sts,st.float().numpy()))
        weight = np.concatenate((weight,w.float().numpy()))
        meds = np.concatenate((meds,med.float().numpy()))
        darks = np.concatenate((darks,dark.float().numpy()))
        rinvs = np.concatenate((rinvs,rinv.float().numpy()))
        model.eval()
        #inputPoints = points.float()
        #inputFeatures = features.float()
        #masks = masks.float()
        with autocast():
            out_tag = model(data.float().to(device))
            #print(out_tag)
            outSoftmax = f.softmax(out_tag,dim=1)
            if i == 0:
                rawOutputs = outSoftmax.cpu().detach().numpy()
            else:
                rawOutputs = np.concatenate((rawOutputs,outSoftmax.cpu().detach().numpy()))
            output_values, output_labels = torch.max(outSoftmax,dim=1) # use output_labels to compare with labels for multiple classification
            output = output_labels.cpu().detach().numpy()
            allOutputs = np.concatenate((allOutputs,output))
            targetLabel = l.numpy()
            targetLabels = np.concatenate((targetLabels,targetLabel))
        # if i == 500:
        #     break
    return targetLabels, allOutputs, np.array(rawOutputs), inputFileIndices, dcorrVars, sts, weight, meds, darks, rinvs

def getROCStuff(label, output, weights=None):
    fpr, tpr, thresholds = roc_curve(label, output, sample_weight=weights)
    auc = roc_auc_score(label, output, sample_weight=weights)
    return fpr, tpr, auc

def getSgBgOutputs(sigCond, bkgCond, output, weights):
    y_Sg = output[sigCond]
    y_Bg = output[bkgCond]
    w_Sg = weights[sigCond]
    w_Bg = weights[bkgCond]

    return y_Sg, y_Bg, w_Sg, w_Bg

def histMakePlot(data,binEdge,color,label,weights=None,alpha=1.0,hatch="",points=False,facecolorOn=True,norm=True):
    hist, bins = np.histogram(data,binEdge,weights=weights,density=norm)
    if points:
        hep.histplot(hist,bins=bins,color=color,label=label,yerr=0,histtype="errorbar",markersize=5)
    else:
        if facecolorOn:
            hep.histplot(hist,bins=bins,color=color,label=label,histtype="fill",hatch=hatch,alpha=alpha)
            hep.histplot(hist,bins=bins,color=color,hatch=hatch,alpha=alpha)
        else:
            hep.histplot(hist,bins=bins,color=color,label=label,hatch=hatch,alpha=alpha)
    return hist

# signal vs. background figure of merit
def fom(S,B):
    return np.sqrt(2 * ( (S+B) * np.log(1+S/B) - S) )

def plotSignificance(cutBins,plotBinEdge,var_train,w_train,inpIndex_train,output_train_tag,tagLabel,varLabel,outf,wpt=0.5):
    foms = []
    binWidth = cutBins[1]-cutBins[0]
    bkgTag = np.logical_and(output_train_tag > wpt,inpIndex_train >= 2)
    sigTag = np.logical_and(output_train_tag > wpt,inpIndex_train <= 1) # inpIndex_train == 1 means using only baseline signal, inpIndex_train <= 1 means all signals
    for j in range(len(cutBins)):
        binVal = cutBins[j]
        if j < len(cutBins)-1:
            varCond = np.absolute(binVal-var_train) < binWidth/2.
        else:
            varCond = var_train > binVal - binWidth/2.
        bkg_var = np.multiply(np.where(bkgTag,1,0),w_train)[varCond]
        sig_var = np.multiply(np.where(sigTag,1,0),w_train)[varCond]
        weighted_sig = np.sum(sig_var)
        weighted_bkg = np.sum(bkg_var)
        foms.append(fom(weighted_sig,weighted_bkg))
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.step(cutBins,foms,where="mid")
    axes = plt.gca()
    plt.text(0.1,0.8,"Max Sig. = {:.2f}".format(np.amax(foms)),transform = axes.transAxes)
    plt.grid()
    # plt.ylim(0,np.nanmax(eff)*1.1)
    plt.ylabel("FOM ( sqrt(2((S+B)*log(1+S/B)-S)) )")
    plt.xlabel('{} (GeV)'.format(varLabel))
    plt.savefig(outf + "/significance_{}.{}".format(varLabel,plotFormat), dpi=fig.dpi, bbox_inches="tight")
    plt.figure(figsize=(12, 8))
    bkgOnly_train = tagLabel == 0
    sigOnly_train = np.logical_not(bkgOnly_train)
    histMakePlot(var_train[sigOnly_train],plotBinEdge,weights=w_train[sigOnly_train],color='xkcd:red',alpha=1.0,label='Sg Train',hatch="//",facecolorOn=False)
    histMakePlot(var_train[bkgOnly_train],plotBinEdge,weights=w_train[bkgOnly_train],color='xkcd:blue',alpha=0.5,label='Bkg Train',hatch="//")
    plt.ylabel("Event")
    plt.xlabel('{} (GeV)'.format(varLabel))
    plt.legend()
    plt.savefig(outf + "/sigVsBkg_{}.{}".format(varLabel,plotFormat), dpi=fig.dpi, bbox_inches="tight")

def plotEffvsVar(binX,var_train,w_train,label_train,output_train_tag,varLabel,outf,dfOut=None,sig=True):
    fig, ax = plt.subplots(figsize=(12, 8))
    binWidth = binX[1]-binX[0]
    for wpt in [0.05,0.1,0.2,0.3,0.5]:
        eff = []
        for j in range(len(binX)):
            binVal = binX[j]
            if j < len(binX)-1:
                varCond = np.absolute(binVal-var_train) < binWidth/2.
            else:
                varCond = var_train > binVal - binWidth/2.
            output_pTC = output_train_tag[varCond]
            label_pTC = label_train[varCond]
            weights_pTC = w_train[varCond]
            output_pTC_wpt = np.where(output_pTC>wpt,1,0)
            if sig:
                ylabel = "Signal Efficiency"
                plotname = outf + "/sigEffVs{}.{}".format(varLabel,plotFormat)
                totalCount = label_pTC
            else:
                ylabel = "Mistag Rate"
                plotname = outf + "/mistagVs{}.{}".format(varLabel,plotFormat)
                totalCount = label_pTC + 1
            weighted_num = np.sum(np.multiply(output_pTC_wpt,weights_pTC))
            weighted_den = np.sum(np.multiply(totalCount,weights_pTC))
            if weighted_den > 0:
                eff.append(weighted_num/weighted_den)
            else:
                eff.append(0)
        binX_ori = np.array(binX)
        eff_ori = np.array(eff)
        binX = binX_ori[eff_ori>0]
        eff = eff_ori[eff_ori>0]
        plt.plot(binX,eff,label="wpt={}".format(wpt))
        if type(dfOut) == pd.core.frame.DataFrame:
            corr = np.corrcoef(binX,eff)
            dfOut["bkgflatness_{}_wpt{}".format(varLabel,wpt)] = [abs(corr[0][1])]
    plt.grid()
    if sig:
        ylabel = "Signal Efficiency"
        plotname = outf + "/sigEffVs{}.{}".format(varLabel,plotFormat)
        plt.ylim(0,1.0)
    else:
        ylabel = "Mistag Rate"
        plotname = outf + "/mistagVs{}.{}".format(varLabel,plotFormat)
        plt.ylim(0,0.4)
    # plt.ylim(0,np.nanmax(eff)*1.1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('{} (GeV)'.format(varLabel))
    plt.legend()
    plt.savefig(plotname, dpi=fig.dpi, bbox_inches="tight")

def plotByBin(binVar,binVarBins,histVar,xlabel,varLab,outDir,plotName,xlim,disLab=False,weights=None,histBinEdge=50,performanceMetrics=None):
    binWidth = binVarBins[1] - binVarBins[0]
    for weightCondition, weights in [["weighted", weights], ["unweighted", np.ones(len(weights))]]:
        fig, ax = plt.subplots(figsize=(12, 8))
        hDataList = []
        for j in range(len(binVarBins)):
            binVal = binVarBins[j]
            if j < len(binVarBins)-1:
                cond = np.absolute(binVal-binVar) < binWidth/2.
                if disLab:
                    lab = '{} = {:.2f}'.format(varLab,binVarBins[j])
                else:
                    lab = '{:.2f} < {} < {:.2f}'.format(binVarBins[j]-binWidth/2.,varLab,binVarBins[j]+binWidth/2)
            else:
                cond = binVar > binVal - binWidth/2.
                if disLab:
                    lab = '{} = {:.2f}'.format(varLab,binVarBins[j])
                else:
                    lab = '{:.2f} < {} < {:.2f}'.format(binVarBins[j]-binWidth/2.,varLab,binVarBins[j]+binWidth/2)
            wVL = weights[cond]
            histVL = histVar[cond]
            if plotName=="SNNpermdark_sig":
                print(plotName)
                print(cond)
                print(np.unique(histVL))
            if len(histVL) > 0:
                histData = histMakePlot(histVL,binEdge=histBinEdge,weights=wVL,color=mplColors[j%len(mplColors)],facecolorOn=False,alpha=0.5,label=lab)
                hDataList.append(histData)
        if len(hDataList) > 1:
            performanceMetrics["avgKS_{}_{}".format(plotName,weightCondition)] = collectiveKS(hDataList)
        ax.set_ylabel('Norm Events')
        ax.set_xlabel(xlabel)
        plt.legend()
        plt.xlim(xlim[0],xlim[1])
        plt.savefig(outDir + "/{}_{}.{}".format(plotName,weightCondition,plotFormat), dpi=fig.dpi, bbox_inches="tight")
        plt.yscale("log")
        # plt.ylim(0.0001,0.2)
        plt.savefig(outDir + "/{}_{}_log.{}".format(plotName,weightCondition,plotFormat), dpi=fig.dpi, bbox_inches="tight")

def plotMultiClassDiscrim(rawOutputs, label, weight, baseClassNum, baseClassLabel, compareClassNum, compareClassLabel, trainType, color, alpha, facecolorOn, hatch, points, compareClassType="Ind"):
    if compareClassType == "Ind":
        compareClassCondition = label == compareClassNum
    else:
        compareClassCondition = label != baseClassNum
    allBaseClassOutputs = rawOutputs[:,baseClassNum] 
    compareClassAsBaseClass = allBaseClassOutputs[compareClassCondition]
    binEdge = np.arange(0,1.001,0.02)
    hist = histMakePlot(compareClassAsBaseClass, binEdge, weights=weight[compareClassCondition], color=color, alpha=alpha, facecolorOn=facecolorOn, hatch=hatch, points=points, label='{} as {} ({})'.format(compareClassLabel,baseClassLabel,trainType))
    return hist, compareClassCondition

def plotMultiClassDiscrimAndROC(rawOutputs_train, label_train, w_train, rawOutputs_test, label_test, w_test, baseClassNum, baseClassLabel, compareClassNum, compareClassLabel, colorDict, outFolder, compareClassType="Ind", performanceMetrics=None):
    # plotting discriminator
    fig, ax = plt.subplots(figsize=(6, 6))
    if compareClassType == "Other":
        compareClassLabel = "Rest"
    histbaseClass_test, baseClassCond_test = plotMultiClassDiscrim(rawOutputs_test, label_test, w_test, baseClassNum, baseClassLabel, baseClassNum, baseClassLabel, "test", colorDict[baseClassLabel], alpha=1.0, facecolorOn=False, hatch="//", points=False, compareClassType="Ind")
    histcompClass_test, compClassCond_test = plotMultiClassDiscrim(rawOutputs_test, label_test, w_test, baseClassNum, baseClassLabel, compareClassNum, compareClassLabel, "test", colorDict[compareClassLabel], alpha=0.5, facecolorOn=True, hatch=None, points=False, compareClassType=compareClassType)
    histbaseClass_train, baseClassCond_train = plotMultiClassDiscrim(rawOutputs_train, label_train, w_train, baseClassNum, baseClassLabel, baseClassNum, baseClassLabel, "train", colorDict[baseClassLabel], alpha=1.0, facecolorOn=True, hatch=None, points=True, compareClassType="Ind")
    histcompClass_train, compClassCond_train = plotMultiClassDiscrim(rawOutputs_train, label_train, w_train, baseClassNum, baseClassLabel, compareClassNum, compareClassLabel, "train", colorDict[compareClassLabel], alpha=1.0, facecolorOn=True, hatch=None, points=True, compareClassType=compareClassType)
    ax.legend(loc='best', fontsize=10, frameon=False)
    print("-"*50)
    print("{} as {}".format(compareClassLabel,baseClassLabel))
    ksComp = kstest(histcompClass_train,histcompClass_test)
    print("histcompClass_train")
    print(histcompClass_train)
    print("histcompClass_test")
    print(histcompClass_test)
    print("ksComp")
    print(ksComp)
    ksBase = kstest(histbaseClass_train,histbaseClass_test)
    print("histbaseClass_train")
    print(histbaseClass_train)
    print("histbaseClass_test")
    print(histbaseClass_test)
    print("ksBase")
    print(ksBase)
    print("-"*50)
    ax.text(0.5,0.75,"KS Test: {:.3f} ({}), {:.3f} ({})".format(ksComp,compareClassLabel,ksBase,baseClassLabel),fontsize=10,transform=ax.transAxes)
    plt.ylabel("Density")
    plt.xlabel("NN Score")
    fig.savefig("{}/prob_{}_as_{}.{}".format(outFolder,compareClassLabel,baseClassLabel,plotFormat), dpi=fig.dpi, bbox_inches="tight")
    performanceMetrics["kstest_discrim_{}_as_{}".format(baseClassLabel,baseClassLabel)] = ksBase
    performanceMetrics["kstest_discrim_{}_as_{}".format(compareClassLabel,baseClassLabel)] = ksComp
    # plotting corresponding ROC
    combinationCond_train = baseClassCond_train | compClassCond_train
    label_train = label_train[combinationCond_train]
    output_train_tag = rawOutputs_train[:,baseClassNum][combinationCond_train]
    w_train = w_train[combinationCond_train]
    combinationCond_test = baseClassCond_test | compClassCond_test
    label_test = label_test[combinationCond_test]
    output_test_tag = rawOutputs_test[:,baseClassNum][combinationCond_test]
    w_test = w_test[combinationCond_test]

    # set baseClass as positive label
    label_train = np.where(label_train==baseClassNum,1,0)
    label_test = np.where(label_test==baseClassNum,1,0)

    fpr_Train, tpr_Train, auc_Train = getROCStuff(label_train, output_train_tag, w_train)
    fpr_Test, tpr_Test, auc_Test = getROCStuff(label_test, output_test_tag, w_test)
    performanceMetrics["auc_train_{}_as_{}".format(compareClassLabel,baseClassLabel)] = auc_Train
    performanceMetrics["auc_test_{}_as_{}".format(compareClassLabel,baseClassLabel)] = auc_Test
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve', pad=45.0)
    plt.plot(fpr_Train, tpr_Train, label="Train (area = {:.3f})".format(auc_Train), color='xkcd:red')
    plt.plot(fpr_Test, tpr_Test, label="Test (area = {:.3f})".format(auc_Test), color='xkcd:black')
    plt.legend(loc='best')
    fig.savefig("{}/roc_plot_{}_as_{}.{}".format(outFolder,compareClassLabel,baseClassLabel,plotFormat), dpi=fig.dpi, bbox_inches="tight")
       
def plotDiscriminator(sigCond_train, bkgCond_train, sigCond_test, bkgCond_test, output_train_tag, w_train, output_test_tag, w_test, outputFolder, dfOut, colorSg, colorBg, bkgLab="Bg"):
    binEdge = np.arange(-0.01, 1.02, 0.02)
    for weightCondition, w_train, w_test in [["weighted", w_train, w_test], ["unweighted", np.ones(len(w_train)), np.ones(len(w_test))]]:
        fig, ax = plt.subplots(figsize=(6, 6))
        y_Train_Sg, y_Train_Bg, w_Train_Sg, w_Train_Bg = getSgBgOutputs(sigCond_train, bkgCond_train, output_train_tag,w_train)
        y_test_Sg, y_test_Bg, w_test_Sg, w_test_Bg = getSgBgOutputs(sigCond_test, bkgCond_test, output_test_tag,w_test)
        ax.set_title('')
        ax.set_ylabel('Norm Events')
        ax.set_xlabel('Discriminator')
        ax.set_xlim(0,1.05)
        h_test_Sg = histMakePlot(y_test_Sg,binEdge,weights=w_test_Sg,color=colorSg,alpha=1.0,label='Sg Test',hatch="//",facecolorOn=False)
        h_test_Bg = histMakePlot(y_test_Bg,binEdge,weights=w_test_Bg,color=colorBg,alpha=0.5,label='{} Test'.format(bkgLab))
        h_Train_Sg = histMakePlot(y_Train_Sg,binEdge,weights=w_Train_Sg,color=colorSg,label='Sg Train',points=True)
        h_Train_Bg = histMakePlot(y_Train_Bg,binEdge,weights=w_Train_Bg,color=colorBg,label='{} Train'.format(bkgLab),points=True)
        dfOut["sigovertrain_{}".format(weightCondition)] = [kstest(h_Train_Sg,h_test_Sg)]
        dfOut["{}overtrain_{}".format(bkgLab,weightCondition)] = [kstest(h_Train_Bg,h_test_Bg)]
        ax.legend(loc='best', frameon=False)
        fig.savefig("{}/discriminator_{}_{}.{}".format(outputFolder,bkgLab,weightCondition,plotFormat), dpi=fig.dpi, bbox_inches="tight")
        # ks test for overtraining

def plotConfusionMatrix(y_true, y_pred, weight, jetClassDict, outFolder, plotLabel=""):
    cmat = confusion_matrix(y_true,y_pred,sample_weight=weight,normalize="true")
    print("label_test",np.unique(y_true,return_counts=True))
    print(cmat)
    # the following condition happens when a jet does not have probability greater than the working point for any of the category
    if len(np.unique(y_pred)) > len(np.unique(y_true)):
        xticklabels = list(np.array(jetClassDict)[:,0]) + ["Unknown"]
        cmat = cmat[:-1] # there is no true unknown jet
    else:
        xticklabels = np.array(jetClassDict)[:,0]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(cmat,annot=True,yticklabels=np.array(jetClassDict)[:,0],xticklabels=xticklabels,cmap='viridis')
    plt.ylabel("Truth")
    plt.xlabel("Predicted")
    plt.savefig(outFolder + "/confusionMatrix{}.{}".format(plotLabel,plotFormat),dpi=fig.dpi, bbox_inches="tight")

def getSigParScores(sp_test, label_test, output_test_tag, w_test, baseline, key):
    sps = np.unique(sp_test)
    allBkgCond = sp_test == 0
    spList = []
    spScores = []
    for sp in sps:
        if (sp == 0):
            continue
        spList.append(sp)
        sigCond = sp_test == sp        
        cond = sigCond | allBkgCond
        fpr_Test, tpr_Test, auc_Test = getROCStuff(label_test[cond], output_test_tag[cond], w_test[cond])
        spScores.append(auc_Test)
    return spList, spScores

def plot2DNNScore(baseline,baselineScore,meds,medScores,otherPara,otherParaScores,otherParaLab,bkgLab,outFolder):
    scores2D = np.full([len(meds),len(otherPara)], np.nan)
    baseYIndex = meds.index(baseline["mMed"])
    baseXIndex = otherPara.index(baseline[otherParaLab])
    for i in range(len(otherPara)):
        scores2D[baseYIndex,i] = otherParaScores[i]
    for i in range(len(meds)):
        scores2D[i,baseXIndex] = medScores[i]
    scores2D[baseYIndex,baseXIndex] = baselineScore
    # meds = np.flip(meds)
    plt.figure()
    plt.imshow(scores2D,vmin=0.5,vmax=1)
    for (i, j), z in np.ndenumerate(scores2D):
        if np.isnan(z):
            z = ""
        else:
            z = '{:0.2f}'.format(z)
        plt.text(j, i, z, ha='center', va='center',fontsize=15)
    if otherParaLab == "mDark":
        plt.xticks(range(len(otherPara)),np.array(otherPara,dtype=int))
    else:
        plt.xticks(range(len(otherPara)),otherPara)
    plt.yticks(range(len(meds)),meds)
    plt.ylabel("mMed")
    plt.xlabel(otherParaLab)
    plt.colorbar()
    plt.savefig(outFolder + "/rocScoreSigP2D{}_{}.{}".format(bkgLab,otherParaLab,plotFormat), bbox_inches="tight")

def plot1DNNvsSigPara(paras,paraScores,paraLab,bkgLab,outFolder,ymin=0.7,ymax=1):
    plt.figure()
    plt.plot(paras,np.around(paraScores,2),marker="o")
    plt.ylabel("NN Score")
    plt.xlabel(paraLab)
    plt.ylim(ymin, ymax)
    plt.savefig(outFolder + "/rocScoreNNvs{}_{}.{}".format(paraLab,bkgLab,plotFormat), bbox_inches="tight")

def main():
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="logs/config_out.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("--outf", type=str, default="logs", help='Name of folder to be used to store outputs')
    #parser.add_argument("--inpf", type=str, default="processedData_nc100", help='Name of the npz input training file')
    parser.add_argument("--model", type=str, default="net.pth", help="Existing model to continue training, if applicable")
    #parser.add_argument("--pIn", action="store_true", help="Plot input variables and their correlation.")
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    modelLocation = "{}/{}".format(args.outf,args.model)
    print("Model location:",modelLocation)

    if not os.path.isdir(args.outf):
        os.mkdir(args.outf)
    # Choose cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        gpuIndex = torch.cuda.current_device()
        print("Using GPU named: \"{}\"".format(torch.cuda.get_device_name(gpuIndex)))

    colorDict = {
        "allBkg":'xkcd:blue',
        "allSig":'xkcd:red',
        "ZJets": "#999999",
        "WJets": "#cccc19",
        "TTJets": "#6666cc",
        "QCD": "#4bd42d",
        "mMed600": "#cd2bcc",
        "baseline": "#cc660d",
        "M-2000_mD-1": "#f2231b",
        "M-2000_mD-100": "#9a27cc",
        "M-2000_r-0p1": "#663303",
        "M-2000_r-0p7": "#f69acc",
        "M-2000_a-low": "#cccccc",
        "M-2000_a-high": "#999910",
        "mMed3000": "#5efdff",
        "SVJ_Dark": "#cd2bcc",
        "SVJ_MixDark": "#cc660d",
        "SVJ": "#f2231b",
        "Rest": "chocolate"
    }

    jetClassDict = [
        ["QCD",      0,0.5,"#4bd42d"],
        ["TTJets",   1,0.5,"#6666cc"],
        ["SVJ", 2,0.5,"#cc660d"] # for t-channel, this is the Dark + MixDark categories
    ]

    performanceMetrics = {}
    # Load dataset
    print('Loading dataset...')
    ds = args.dataset
    hyper = args.hyper
    ft = args.features
    tr = args.training
    trainData, trainNonTrainingInfo, trainData_nonUniform, trainNonTrainingInfo_nonUniform, valData, valNonTrainingInfo, testData, testNonTrainingInfo = getDataset(ds.path, ds.signal, ds.background, ds.sample_fractions, ft.eventVariables, ft.jetVariables, ft.dcorrVar, tr.weights, ft.numOfJetsToKeep,ds.flatMET)
    train = RootDataset(trainData_nonUniform, trainNonTrainingInfo_nonUniform)
    test = RootDataset(testData, testNonTrainingInfo)
    # Build model
    model = DNN(n_var=len(train[0]["data"]), n_layers=hyper.num_of_layers, n_nodes=hyper.num_of_nodes, n_outputs=hyper.num_classes, drop_out_p=hyper.dropout).to(device=device)
    #model = DNN_GRF(n_var=len(varSet), n_layers_features=hyper.num_of_layers_features, n_layers_tag=hyper.num_of_layers_tag, n_layers_pT=hyper.num_of_layers_pT, n_nodes=hyper.num_of_nodes, n_outputs=2, n_pTBins=hyper.n_pTBins, drop_out_p=hyper.dropout).to(device=device)
    model = copy.deepcopy(model)
    print("Loading model from file " + modelLocation)
    model.load_state_dict(torch.load(modelLocation,map_location=device))
    model.eval()
    model.to(device)

    label_test, output_test_tag, rawOutputs_test, inpIndex_test, dcorrVar_test, st_test, w_test, med_test, dark_test, rinv_test = getNNOutput(test, model, device)
    ## sanity check: calculating disco and pearson correlation for test dataset
    # normedweight = np.ones_like(output_test_tag)
    # print("all unique label_test",np.unique(label_test))
    # print("all unique output_test_tag",np.unique(output_test_tag))
    # print("rawOutputs_test",rawOutputs_test.shape)
    # nnScore_test = torch.Tensor(np.round(rawOutputs_test[:,2],4))
    # dcorrVar_test = torch.Tensor(np.round(dcorrVar_test,4))
    # st_test = torch.Tensor(np.round(st_test,4))
    # normedweight = torch.Tensor(normedweight)
    # for i in range(3):
    #     cond = label_test == i
    #     print(f"label {i}")
    #     discoVal = distance_corr(1, nnScore_test[cond][:1000], dcorrVar_test[cond][:1000], normedweight[cond][:1000])
    #     print(f"discoVal (MET): {discoVal}")
    #     corr = np.corrcoef(nnScore_test[cond],dcorrVar_test[cond])[0,1]
    #     print(f"Pearson corr (MET): {corr}")
    #     discoVal = distance_corr(1, nnScore_test[cond][:1000], st_test[cond][:1000], normedweight[cond][:1000])
    #     print(f"discoVal (ST): {discoVal}")
    #     corr = np.corrcoef(nnScore_test[cond],st_test[cond])[0,1]
    #     print(f"Pearson corr (ST): {corr}")
    #     print()

    # 2D plots of ROC vs signal parameters
    ## get auc for just baseline
    baseline = {
        "mMed": 2000, # 2000 for t-channel, 3000 for s-channel
        "mDark": 20,
        "rinv": 0.3
    }
    # there is floating point precision problem if we do not round the numbers
    med_test = np.round(med_test)
    dark_test = np.round(dark_test,1)
    rinv_test = np.round(rinv_test,1)
    bkgLabDict = {
        "QCD":0,
        "TTJets":1
    } 

    for bkgLab, bkgCode in bkgLabDict.items():
        print(bkgLab, bkgCode)
        bkgSigCond = (label_test == bkgCode) | (label_test == 2)
        med_testBkgSig = med_test[bkgSigCond]
        dark_testBkgSig = dark_test[bkgSigCond]
        rinv_testBkgSig = rinv_test[bkgSigCond]

        output_test_tagBkgSig = rawOutputs_test[:,2][bkgSigCond]
        label_testBkgSig = np.where(label_test[bkgSigCond]==2,1,0)
        w_testBkgSig = w_test[bkgSigCond]

        baselineCond = (med_testBkgSig == baseline["mMed"]) & (np.round(dark_testBkgSig,1) == baseline["mDark"]) & (np.round(rinv_testBkgSig,1) == baseline["rinv"])       
        cond = baselineCond | (med_testBkgSig == 0) # background has 0 for all the signal parameters
        print("baseline_qcd_cond in loop",np.sum(cond))
        fpr_baseline_Test, tpr_baseline_Test, baselineScore = getROCStuff(label_testBkgSig[cond], output_test_tagBkgSig[cond], w_testBkgSig[cond])

        meds, medScores = getSigParScores(med_testBkgSig, label_testBkgSig, output_test_tagBkgSig, w_testBkgSig, "mMed", baseline)
        darks, darkScores = getSigParScores(dark_testBkgSig, label_testBkgSig, output_test_tagBkgSig, w_testBkgSig, "mDark", baseline)
        rinvs, rinvScores = getSigParScores(rinv_testBkgSig, label_testBkgSig, output_test_tagBkgSig, w_testBkgSig, "rinv", baseline)    

        medScores[meds.index(baseline["mMed"])] = baselineScore
        darkScores[darks.index(baseline["mDark"])] = baselineScore
        rinvScores[rinvs.index(baseline["rinv"])] = baselineScore

        plot1DNNvsSigPara(meds,medScores,"mMed",bkgLab,args.outf,ymin=0.7,ymax=0.98)
        plot1DNNvsSigPara(darks,darkScores,"mDark",bkgLab,args.outf,ymin=0.7,ymax=0.98)
        plot1DNNvsSigPara(rinvs,rinvScores,"rinv",bkgLab,args.outf,ymin=0.7,ymax=0.98)

        plot2DNNScore(baseline,baselineScore,meds,medScores,darks,darkScores,"mDark",bkgLab,args.outf)
        plot2DNNScore(baseline,baselineScore,meds,medScores,rinvs,rinvScores,"rinv",bkgLab,args.outf)

    ## Correlation between dcorrVar and NN Score
    for mainClass in jetClassDict:
        mainClassName = mainClass[0]
        mainClassNum = mainClass[1]
        classCond = label_test == mainClassNum 
        dcorrVar_test_class = dcorrVar_test[classCond]
        w_test_class = w_test[classCond]
        for asClass in jetClassDict:
            asClassName = asClass[0]
            asClassNum = asClass[1]
            output_test_tag_class = rawOutputs_test[classCond][:,asClassNum]
            plotByBin(binVar=dcorrVar_test_class,binVarBins = np.arange(0,1000,200),histVar=output_test_tag_class,xlabel="NN Score",varLab="{}".format(ft.dcorrVar),outDir=args.outf,plotName="SNNper{}_{}_as_{}".format(ft.dcorrVar,mainClassName,asClassName),xlim=[0,1.02],weights=w_test_class,histBinEdge=np.arange(-0.01, 1.02, 0.05),performanceMetrics=performanceMetrics)
            plotByBin(binVar=output_test_tag_class,binVarBins = np.arange(0.1,1,0.2),histVar=dcorrVar_test_class,xlabel="{}".format(ft.dcorrVar),varLab="SNN",outDir=args.outf,plotName="{}perSNN_{}_as_{}".format(ft.dcorrVar,mainClassName,asClassName),xlim=[0,1500],weights=w_test_class,histBinEdge=np.arange(-50,1501,50),performanceMetrics=performanceMetrics)

    # making confusion matrix to see which jet categories are harder to classify
    print("output_test_tag",np.unique(output_test_tag,return_counts=True))
    ## this confusion_matrix assumes that the category with the highest probability is the predicted category
    plotConfusionMatrix(label_test, output_test_tag, w_test, jetClassDict, args.outf)
    plotConfusionMatrix(label_test, output_test_tag, np.ones(len(w_test)), jetClassDict, args.outf, "_unweighted")
    # confusion matrix based on the discrimination distribution between each category and the rest of the categories.
    workingPts = np.array(jetClassDict)[:,2].astype(float)
    wpt_outputs = []
    for rawOutput in rawOutputs_test:
        candidatePositions = np.where(rawOutput > workingPts)[0]
        if len(candidatePositions) == 0:
            candidate = len(rawOutput)
        else:
            candidateProbs = np.take(rawOutput,candidatePositions)
            candidate = candidatePositions[np.argmax(candidateProbs)]
        wpt_outputs.append(candidate)
    plotConfusionMatrix(label_test, wpt_outputs, w_test, jetClassDict, args.outf, plotLabel="WorkingPts")
    plotConfusionMatrix(label_test, wpt_outputs, np.ones(len(w_test)), jetClassDict, args.outf, plotLabel="WorkingPts_unweighted")
    # making discrimination and ROC plots for different jet categories for the OvO and OvR cases
    label_train, output_train_tag, rawOutputs_train, inpIndex_train, dcorrVar_train, st_train, w_train, med_train, dark_train, rinv_train = getNNOutput(train, model, device)
    combos = list(itertools.product(range(len(jetClassDict)),range(len(jetClassDict))))
    for combo in combos:
        base = combo[0]
        compare = combo[1]
        baseClassLabel = jetClassDict[base][0]
        baseClassNum = jetClassDict[base][1]
        compareClassLabel = jetClassDict[compare][0]
        compareClassNum = jetClassDict[compare][1]
        if base == compare:
            compareClassType = "Other"
        else:
            compareClassType = "Ind"
        plotMultiClassDiscrimAndROC(rawOutputs_train, label_train, w_train, rawOutputs_test, label_test, w_test, baseClassNum, baseClassLabel, compareClassNum, compareClassLabel, colorDict, args.outf, compareClassType=compareClassType, performanceMetrics=performanceMetrics)

    np.savez("{}/performanceMetrics.npz".format(args.outf),**performanceMetrics)
    print(performanceMetrics)

if __name__ == "__main__":
    main()
