import numpy as np
import uproot as up
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import torch.utils.data as udata
import torch
import pandas as pd
from torchvision import transforms

def npAppend(array1,array2):
    if np.any(array1) != True:
        array1 = [array2]
    else:
        array1 = np.concatenate((array1,[array2]))

def getBranch(f,tree,variable,branches,branchList):
    branch = f[tree].pandas.df(variable)
    branch = branch.head(len(branches))
    branchList.append(branch)

def getPara(fileName,paraName,paraList,branches,key):
    paravalue = 0
    if key == "signal":
        ind = fileName.find(paraName)
        fnCut = fileName[ind:]
        indUnd = fnCut.find("_")
        paravalue = fnCut[len(paraName)+1:indUnd]
        if paraName == "alpha":
            if paravalue == "low":
                paravalue = 1
            elif paravalue == "peak":
                paravalue = 2
            elif paravalue == "high":
                paravalue = 3
        else:
            paravalue = float(paravalue)
    paraList += [paravalue]*len(branches)

def normalize(df):
    return (df-df.mean())/df.std()

def get_all_vars(inputFolder, samples, variables, pTBins, uniform, mT, weight, tree="tree"):
    dSets = []
    signal = []
    mcType =[] # 0 = signals other than baseline, 1 = baseline signal, 2 = QCD, 3 = TTJets
    pTLab = np.array([])
    pTs = []
    mTs = []
    mMeds = []
    mDarks = []
    rinvs = []
    alphas = []
    weights = []
    for key,fileList in samples.items():
        nsigfiles = len(samples["signal"])
        nbkgfiles = len(samples["background"])
        for fileName in fileList:
            print(fileName)
            f = up.open(inputFolder  + fileName + ".root")
            branches = f[tree].pandas.df(variables)
            if key == "signal":
                jetCatBranch = f[tree].pandas.df("jCsthvCategory")
                darkCon = ((jetCatBranch["jCsthvCategory"] == 3) | (jetCatBranch["jCsthvCategory"] == 9))
                # print(jetCatBranch['jCsthvCategory'].value_counts())
                branches = branches[darkCon]
            branches.replace([np.inf, -np.inf], np.nan, inplace=True)
            branches = branches.dropna()
            numEvent = len(branches)
            maxNum = 20000 # 10735 t-channel # 10031 s-channel # 29761 pair production
            numEvent = maxNum
            if key == "background":
                numEvent = int(nsigfiles*maxNum/nbkgfiles)
            branches = branches.head(numEvent) #Hardcoded only taking ~30k events per file while we setup the code; should remove this when we want to do some serious trainings
            print(len(branches))
            # branches = branches.head(10000)
            dSets.append(branches)
            getBranch(f,tree,uniform,branches,pTs)
            getBranch(f,tree,mT,branches,mTs)
            getBranch(f,tree,weight,branches,weights)
            # getPara(fileName,"mZprime",mMeds,branches,key)
            getPara(fileName,"mMed",mMeds,branches,key) # use this for t-channel
            getPara(fileName,"mDark",mDarks,branches,key)
            getPara(fileName,"rinv",rinvs,branches,key)
            getPara(fileName,"alpha",alphas,branches,key)
            # get the pT label based on what pT bin the jet pT falls into
            branch = f[tree].pandas.df(uniform)
            branch = branch.head(len(branches)).to_numpy().flatten()
            pTLabel = np.digitize(branch,pTBins) - 1.0
            pTLab = np.append(pTLab,pTLabel)
            if key == "signal":
                signal += list([0, 1] for _ in range(len(branches)))
                if fileName == "tree_SVJ_mZprime-3000_mDark-20_rinv-0.3_alpha-peak_MC2017":
                    mcType += [1] * len(branches)
                else:
                    mcType += [0] * len(branches)
            else:
                signal += list([1, 0] for _ in range(len(branches)))
                if "QCD" in fileName:
                    mcType += [2] * len(branches)
                else:
                    mcType += [3] * len(branches)
            print("Number of Dark AK8Jets",len(branches))
    mcType = np.array(mcType)
    mMed = np.array(mMeds)
    mDark = np.array(mDarks)
    rinv = np.array(rinvs)
    alpha = np.array(alphas)
    dataSet = pd.concat(dSets)
    print(dataSet.head())
    dfmean = dataSet.mean()
    dfstd = dataSet.std()
    dataSet = normalize(dataSet)
    pT = pd.concat(pTs)
    mT = pd.concat(mTs)
    weight = pd.concat(weights)
    return [dataSet,signal,mcType,pTLab,pT,mT,weight,mMed,mDark,rinv,alpha,dfmean,dfstd]

def get_sizes(l, frac=[0.8, 0.1, 0.1]):
    if sum(frac) != 1.0: raise ValueError("Sum of fractions does not equal 1.0")
    if len(frac) != 3: raise ValueError("Need three numbers in list for train, test, and val respectively")
    train_size = int(frac[0]*l)
    test_size = int(frac[1]*l)
    val_size = l - train_size - test_size
    return [train_size, test_size, val_size]

class RootDataset(udata.Dataset):
    def __init__(self, inputFolder, root_file, variables, pTBins, uniform, mT, weight):
        dataSet, signal, mcType, pTLab, pTs, mTs, weights, mMeds, mDarks, rinvs, alphas,dfmean,dfstd = get_all_vars(inputFolder, root_file, variables, pTBins, uniform, mT, weight)
        self.root_file = root_file
        self.variables = variables
        self.uniform = uniform
        self.weight = weight
        self.vars = dataSet.astype(float).values
        self.signal = signal
        self.mcType = mcType
        self.pTLab = pTLab
        self.pTs = pTs.astype(float).values
        self.mTs = mTs.astype(float).values
        self.weights = weights.astype(float).values
        self.mMeds = mMeds
        self.mDarks = mDarks
        self.rinvs = rinvs
        self.alphas = alphas
        self.normMean = np.array(dfmean)
        self.normstd = np.array(dfstd)
        print("Number of events:", len(self.signal))

    #def get_arrays(self):
    #    return np.array(self.signal), torch.from_numpy(self.vars.astype(float).values.copy()).float().squeeze(1)

    def __len__(self):
        return len(self.vars)

    def __getitem__(self, idx):
        data_np = self.vars[idx].copy()
        label_np = np.zeros(1, dtype=np.long).copy()
        mcType_np = np.array([np.long(self.mcType[idx])]).copy()
        pTLab_np = np.array([np.long(self.pTLab[idx])]).copy()
        pTs_np = self.pTs[idx].copy()
        mTs_np = self.mTs[idx].copy()
        weights_np = self.weights[idx].copy()
        mMeds_np = np.array([self.mMeds[idx]]).copy()
        mDarks_np = np.array([self.mDarks[idx]]).copy()
        rinvs_np = np.array([self.rinvs[idx]]).copy()
        alphas_np = np.array([np.long(self.alphas[idx])]).copy()

        if self.signal[idx][1]:
            label_np += 1

        data  = torch.from_numpy(data_np)
        # print("Data inside getitem")
        # print(data)
        label = torch.from_numpy(label_np)
        mcType = torch.from_numpy(mcType_np)
        pTLab = torch.from_numpy(pTLab_np)
        pTs = torch.from_numpy(pTs_np).float()
        mTs = torch.from_numpy(mTs_np).float()
        weights = torch.from_numpy(weights_np).float()
        mMeds = torch.from_numpy(mMeds_np).float()
        mDarks = torch.from_numpy(mDarks_np).float()
        rinvs = torch.from_numpy(rinvs_np).float()
        alphas = torch.from_numpy(alphas_np)
        return label, data, mcType, pTLab, pTs, mTs, weights, mMeds, mDarks, rinvs, alphas

if __name__=="__main__":
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="configs/C1.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    dSet = args.dataset
    sigFiles = dSet.signal
    inputFiles = dSet.background
    inputFiles.update(sigFiles)
    print(inputFiles)
    varSet = args.features.train
    print(varSet)
    pTBins = args.hyper.pTBins
    print(pTBins)
    uniform = args.features.uniform
    mTs = args.features.mT
    weights = args.features.weight
    dataset = RootDataset(dSet.path, inputFiles, varSet, pTBins, uniform, mTs, weights)
    sizes = get_sizes(len(dataset), dSet.sample_fractions)
    train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))
    loader = udata.DataLoader(dataset=train, batch_size=train.__len__(), num_workers=0)
    l, d, mct, pl, p, m, w, med, dark, rinv, alpha = next(iter(loader))
    labels = l.squeeze(1).numpy()
    mcType = mct.squeeze(1).numpy()
    pTLab = pl.squeeze(1).numpy()
    data = d.float().numpy()
    pTs = p.squeeze(1).float().numpy()
    mTs = m.squeeze(1).float().numpy()
    weights = w.squeeze(1).float().numpy()
    meds = med.squeeze(1).float().numpy()
    darks = dark.squeeze(1).float().numpy()
    rinvs = rinv.squeeze(1).float().numpy()
    alphas = alpha.squeeze(1).numpy()
    print("labels:", labels)
    print("Number of signals: ",len(labels[labels==1]))
    print("Number of backgrounds: ",len(labels[labels==0]))

    # put the following block of code into a function, make sure the input features and input points are correct.
    evtNumIndex = varSet.index("jCstEvtNum")
    fJetNumIndex = varSet.index("jCstJNum")
    etaIndex = varSet.index("jCstEta")
    phiIndex = varSet.index("jCstPhi")
    hvIndex = varSet.index("jCsthvCategory")
    evtNumColumn = data[:,evtNumIndex]
    fJetNumColumn = data[:,fJetNumIndex]
    evtFjetCom = []
    for com in np.transpose([evtNumColumn,fJetNumColumn]):
        alreadyExist = False
        for u in evtFjetCom:
            if (u[0] == com[0]) and (u[1] == com[1]):
                alreadyExist = True
                break
        if alreadyExist:
            continue
        else:
            evtFjetCom.append(com)

    inputPoints = np.array([])
    inputFeatures = np.array([])
    totalEntry = 100 # following what the particleNet example did
    for evtNum,fJNum in evtFjetCom: # somehow np.unique flipped the two entries in each row??
        sameJetConstData = data[(evtNumColumn == evtNum) & (fJetNumColumn == fJNum)] # getting values for constituents in the same jet
        sameJetConstDataTr = np.transpose(sameJetConstData)
        if totalEntry > sameJetConstDataTr.shape[1]:
            paddedJetConstData = np.pad(sameJetConstDataTr,((0,0),(0,totalEntry-sameJetConstDataTr.shape[1])), 'constant', constant_values=0)
        else:
            paddedJetConstData = sameJetConstDataTr[:,:totalEntry]
        eachJetPoints = np.array([paddedJetConstData[etaIndex],paddedJetConstData[phiIndex]])
        eachJetFeatures = []
        for i in range(paddedJetConstData.shape[0]):
            if i in [evtNumIndex,fJetNumIndex,hvIndex]:
                continue
            else:
                if len(paddedJetConstData[i]) != 100:
                    print(len(paddedJetConstData[i]))
                eachJetFeatures.append(paddedJetConstData[i])
        if np.any(inputPoints) != True:
            inputPoints = [eachJetPoints]
        else:
            inputPoints = np.concatenate((inputPoints,[eachJetPoints]))
        npAppend(inputPoints,eachJetPoints)
        npAppend(inputFeatures,eachJetFeatures)
        if np.any(inputFeatures) != True:
            inputFeatures = [eachJetFeatures]
        else:
            inputFeatures = np.concatenate((inputFeatures,[eachJetFeatures]))

    print("inputPoints:", inputPoints.shape)
    print("inputFeatures:", inputFeatures.shape)
    print("inputData:", data)
    print("inputData shape:", data.shape)
    print("Input has nan:",np.isnan(np.sum(data)))
    print("inputMean:", np.mean(data,axis=0))
    print("inputSTD:", np.std(data,axis=0))
    print("mcType:", mcType)
    print("pTLab:", pTLab)
    print("mT:", mTs)
    print("pT:", pTs)
    print("weights", weights)
    print("meds:", np.unique(meds))
    print("darks:", np.unique(darks))
    print("rinvs:", np.unique(rinvs))
    print("alphas", np.unique(alphas))
