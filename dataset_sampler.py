import numpy as np
import uproot as up
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import torch.utils.data as udata
import torch
import pandas as pd
from tqdm import tqdm
import awkward as ak
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None 

def getCondition(label,key):
    return label == key

def getReferencePtHist(normedData,nonTrainingInfo,randomGenerator):
    label = normedData["label"]
    pT = nonTrainingInfo["dcorrVar"]
    weight = nonTrainingInfo["w"]
    pTBin = np.arange(0,2000,20)
    useLab = 2
    fracToUse = 1.0
    indices = np.where(getCondition(label,useLab))[0]
    useInd = randomGenerator.choice(indices,int(len(indices)*fracToUse),replace=False)
    referencePtHist, bins = np.histogram(np.take(pT,useInd,axis=0),pTBin)
    return referencePtHist,pTBin,useInd

def getSigJetIndices(referenceHist,pTBin,label,key,pT):
    digitpT = np.digitize(pT,pTBin) - 1
    jetCond = getCondition(label,key)
    finalIndices = []
    for j in range(len(referenceHist)):
        nJ_j = referenceHist[j]
        allJ_j_pos = np.where((digitpT == j) & (jetCond))[0]
        if (nJ_j == 0) or (len(allJ_j_pos) == 0):
            continue
        if len(allJ_j_pos) > nJ_j:
            finalIndices += list(randomGenerator.choice(allJ_j_pos,nJ_j,replace=False))
        else:
            finalIndices += list(randomGenerator.choice(allJ_j_pos,nJ_j,replace=True))
    return finalIndices

# this not only samples the background jets in a way that match some reference histogram, it also makes sure that the proportion of jets in each pT bin matches the actual proportion of jets from each subsample in that bin
def getBkgJetIndices(referenceHist,pTBin,key,inputFileNames,nonTrainingInfo,randomGenerator):
    inputFileIndices = nonTrainingInfo["inputFileIndices"]
    pT = nonTrainingInfo["dcorrVar"]
    weight = nonTrainingInfo["w"]
    weightedHistList = []
    histFileConList = []
    finalIndices = []
    for i in range(len(inputFileNames)):
        inFile = inputFileNames[i]
        if key in inFile:
            cond = inputFileIndices == i
            weightedHist, bins = np.histogram(pT[cond],pTBin,weights=weight[cond])
            weightedHistList.append(weightedHist)
            histFileConList.append(cond)
    totalWeightedHist = np.sum(weightedHistList,axis=0)
    numberOfJetsToKeepList = []
    for weightedHist in weightedHistList:
        numberOfJetsToKeepList.append((weightedHist/totalWeightedHist) * referenceHist)
    digitpT = np.digitize(pT,pTBin) - 1
    for i in range(len(histFileConList)):
        histFileCon = histFileConList[i]
        numberOfJetsToKeep = numberOfJetsToKeepList[i]
        for j in range(len(numberOfJetsToKeep)):
            nJ_j = int(np.nan_to_num(numberOfJetsToKeep[j],posinf=0, neginf=0))
            allJ_j_pos = np.where((digitpT == j) & (histFileCon))[0]
            if (nJ_j == 0) or (len(allJ_j_pos) == 0):
                continue
            if len(allJ_j_pos) > nJ_j:
                finalIndices += list(randomGenerator.choice(allJ_j_pos,nJ_j,replace=False))
            else:
                finalIndices += list(randomGenerator.choice(allJ_j_pos,nJ_j,replace=True))
    return finalIndices

def get_sizes(l, frac=[0.8, 0.1, 0.1]):
    if sum(frac) != 1.0: raise ValueError("Sum of fractions does not equal 1.0")
    if len(frac) != 3: raise ValueError("Need three numbers in list for train, test, and val respectively")
    train_size = int(frac[0]*l)
    test_size = int(frac[1]*l)
    val_size = l - train_size - test_size
    return [train_size, test_size, val_size]

def normalize(data):
    copyData = data.copy(deep=True)
    for column in copyData.columns:
        if "jEtaAK8" in column:
            copyData[column] = abs(copyData[column])
        elif column in ["METrHT_pt30","dEtaj12AK8","dRJ12AK8","dPhiMinjMETAK8"]:
            continue
        elif "dPhijMETAK8" in column:
            continue
        elif "jPhiAK8" in column:
            continue
        elif "nnOutput" in column:
            continue
        else:
            copyData[column] = np.log(copyData[column])
    return(copyData)

def getPara(fileName,paraName):
    paravalue = 0
    if "SVJ" in fileName:
        ind = fileName.find(paraName)
        fnCut = fileName[ind:]
        indUnd = fnCut.find("_")
        paravalue = fnCut[len(paraName)+1:indUnd]
        if paraName == "rinv":
            paravalue = float(paravalue.replace("p","."))
        else:
            paravalue = float(paravalue)
    return paravalue

def scale_weight(inputFileIndices_subset,inputFileIndices,nonTrainingInfo):
    uniInput, uniCount = np.unique(inputFileIndices,return_counts=True)
    uniInput_sub, uniCount_sub = np.unique(inputFileIndices_subset,return_counts=True)
    scales = uniCount/uniCount_sub
    for i in range(len(uniInput_sub)):
        uIn_sub = uniInput_sub[i]
        scale = scales[i]
        nonTrainingInfo['w'][inputFileIndices_subset == uIn_sub] *= scale

def open_data(path, dataFiles, fileDictForOffset, eventVar, jetVar, dcorrVar, weights, numOfJetsToKeep):
    vars = None
    names = None
    inputFileNames = []
    labelShift = len(fileDictForOffset)
    fileIndexShift = 0
    for key,files in fileDictForOffset.items(): 
        fileIndexShift += len(files)
    fileIndex = 0
    for i, key in enumerate(dataFiles):
        fileList = dataFiles[key]
        for fileName in fileList:
            f = up.open(path  + fileName + ".root")
            eventvar = f["tree"].arrays(eventVar,  library="pd")#.head(100)
            jetvar = f["tree"].arrays(jetVar,  library="pd")#.head(100)
            regex = ""
            for nj in range(numOfJetsToKeep):
                regex += '\['+str(nj)+'\]|'
            jetvar = jetvar.filter(regex=regex[:-1])
            var = pd.concat([eventvar , jetvar ],axis=1)
            var["label"] = [i+labelShift]*len(var)
            vars = pd.concat([vars, var]) if vars is not None else var
            name = pd.DataFrame()
            name["dcorrVar"] = f["tree"].arrays(dcorrVar,  library="pd") 
            name["st"] = f["tree"].arrays("st",  library="pd") 
            name["w"]     = f["tree"].arrays(weights,  library="pd") 
            name["mMed"]  = [getPara(fileName,  "mMed")]*len(var)
            name["mDark"] = [getPara(fileName, "mDark")]*len(var)
            name["rinv"]  = [getPara(fileName,  "rinv")]*len(var)
            name["inputFileIndices"] = [fileIndexShift+fileIndex]*len(var)
            fileIndex += 1
            names = pd.concat([names, name]) if names is not None else name
            inputFileNames.append(fileName)
    return vars, names, inputFileNames

def uniformMETSampling(normedData,nonTrainingInfo,inputFileNames):
    randomGenerator = np.random.default_rng(888)
    referenceHist,pTBin,referenceIndices = getReferencePtHist(normedData,nonTrainingInfo,randomGenerator)
    QCDFinalIndices = getBkgJetIndices(referenceHist,pTBin,"QCD",inputFileNames,nonTrainingInfo,randomGenerator)
    TTJetsFinalIndices = getBkgJetIndices(referenceHist,pTBin,"TTJets",inputFileNames,nonTrainingInfo,randomGenerator)
    finalIndices = QCDFinalIndices + TTJetsFinalIndices + list(referenceIndices)
    finalData = normedData.iloc[finalIndices]
    finalNonTrainingInfo = nonTrainingInfo.iloc[finalIndices]
    return finalData, finalNonTrainingInfo

def getDataset(path, sigFiles, bkgFiles, sample_fractions, eventVar, jetVar, dcorrVar, weights, numOfJetsToKeep,flatMET=False):
    bkgData, bkgNames, bkgInputFileNames = open_data(path, bkgFiles, {}, eventVar, jetVar, dcorrVar, weights, numOfJetsToKeep)
    sigData, sigNames, sigInputFileNames = open_data(path, sigFiles, bkgFiles, eventVar, jetVar, dcorrVar, weights, numOfJetsToKeep)
    allData = pd.concat([bkgData,sigData])
    # normalizing inputs
    inputVars = list(allData.columns)
    inputVars.remove("label")
    inputData = allData[inputVars]
    # normedData= pd.concat([inputData,allData["label"]],axis=1)
    normedInputData = normalize(inputData)
    normedData= pd.concat([normedInputData,allData["label"]],axis=1)
    # this part needs to be updated so that bkgNames and sigNames will be masked properly too. As of now, it's not doing anything anyway.
    # normedData.replace([np.inf, -np.inf], np.nan, inplace=True)
    # normedData.dropna(inplace=True)
    # print("After drop na, normedData length: ",len(normedData))
    # reference histogram is the sum of all SVJ signals' MET
    nonTrainingInfo = pd.concat([bkgNames, sigNames])
    inputFileNames = bkgInputFileNames + sigInputFileNames
    
    # split between train and the others
    trainFraction = sample_fractions[0]
    otherFraction = 1 - trainFraction
    train_indices, other_indices, train_inputFileIndices, other_inputFileIndices = train_test_split(range(len(normedData)), nonTrainingInfo["inputFileIndices"], test_size=otherFraction, random_state=42,stratify=nonTrainingInfo["inputFileIndices"])
    # split between test and validation
    testFraction = sample_fractions[1]
    testRelativeFraction = testFraction/otherFraction
    test_indices, validation_indices = train_test_split(other_indices, test_size=testRelativeFraction, random_state=42,stratify=other_inputFileIndices)
    trainData_nonUniform = normedData.iloc[train_indices]
    trainNonTrainingInfo_nonUniform = nonTrainingInfo.iloc[train_indices]
    testData = normedData.iloc[test_indices]
    testNonTrainingInfo = nonTrainingInfo.iloc[test_indices]
    if flatMET:
        trainData, trainNonTrainingInfo = uniformMETSampling(normedData.iloc[train_indices],nonTrainingInfo.iloc[train_indices],inputFileNames)
        valData, valNonTrainingInfo = uniformMETSampling(normedData.iloc[validation_indices],nonTrainingInfo.iloc[validation_indices],inputFileNames)
    else:
        trainData = trainData_nonUniform
        trainNonTrainingInfo = trainNonTrainingInfo_nonUniform
        valData = normedData.iloc[validation_indices]
        valNonTrainingInfo = nonTrainingInfo.iloc[validation_indices]
    # weights are used only for making validation plots. Since only train and test sets are used for the plotting, we do not need to scale the weights for the val data.
    scale_weight(trainNonTrainingInfo["inputFileIndices"],nonTrainingInfo["inputFileIndices"],trainNonTrainingInfo)
    scale_weight(testNonTrainingInfo["inputFileIndices"],nonTrainingInfo["inputFileIndices"],testNonTrainingInfo)
    print("trainData size:",len(trainData))
    print("valData size:",len(valData))
    print("testData size:",len(testData))
    return trainData, trainNonTrainingInfo, trainData_nonUniform, trainNonTrainingInfo_nonUniform, valData, valNonTrainingInfo, testData, testNonTrainingInfo

if __name__=="__main__":
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="configs/C1_EventLevel.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    ds = args.dataset
    ft = args.features
    tr = args.training
    trainData, trainNonTrainingInfo, valData, valNonTrainingInfo, testData, testNonTrainingInfo = getDataset(ds.path, ds.signal, ds.background, ds.sample_fractions, ft.eventVariables, ft.jetVariables, ft.dcorrVar, tr.weights, ft.numOfJetsToKeep)
    print("trainData",len(trainData))
    print("valData",len(valData))
    print("testData",len(testData))
    