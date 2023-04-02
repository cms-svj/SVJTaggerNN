import numpy as np
import uproot as up
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import torch.utils.data as udata
import torch
import pandas as pd
from tqdm import tqdm

def get_sizes(l, frac=[0.8, 0.1, 0.1]):
    if sum(frac) != 1.0: raise ValueError("Sum of fractions does not equal 1.0")
    if len(frac) != 3: raise ValueError("Need three numbers in list for train, test, and val respectively")
    train_size = int(frac[0]*l)
    test_size = int(frac[1]*l)
    val_size = l - train_size - test_size
    return [train_size, test_size, val_size]

class RootDataset(udata.Dataset):
    def __init__(self, path, sigFiles, bkgFiles, eventVar, jetVar):
        self.eventVar = eventVar
        self.jetVar = jetVar
        bkgData, bkgNames = self.open_data(path, bkgFiles, 0)
        sigData, sigNames = self.open_data(path, sigFiles, len(bkgFiles))
        self.allData = pd.concat([bkgData,sigData])
        self.names = pd.concat([bkgNames, sigNames])

    def getPara(self, fileName,paraName):
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

    def open_data(self, path, dataFiles, labelShift):
        vars = None
        names = None
        for i, key in enumerate(dataFiles):
            fileList = dataFiles[key]
            for fileName in fileList:
                f = up.open(path  + fileName + ".root")
                print(fileName)
                var = f["tree"].arrays(self.eventVar + self.jetVar,  library="pd")#.head(10000)
                var = var.filter(regex='[0]|[1]|[2]')
                var["label"] = [i+labelShift]*len(var)
                vars = pd.concat([vars, var]) if vars is not None else var

                name = pd.DataFrame()
                name["w"]     = [1.0]*len(var) 
                name["mMed"]  = [self.getPara(fileName,  "mMed")]*len(var)
                name["mDark"] = [self.getPara(fileName, "mDark")]*len(var)
                name["rinv"]  = [self.getPara(fileName,  "rinv")]*len(var)
                names = pd.concat([names, name]) if names is not None else name

        return vars, names

    def __len__(self):
        return len(self.allData)

    def __getitem__(self, idx):
        row = self.allData.iloc[idx]
        data, label = row.iloc[:-1], row.iloc[-1]
        return dict(
            data=torch.tensor(data),
            label=torch.tensor(label, dtype=torch.long),
            w = torch.tensor(self.names["w"].iloc[idx]),
            mMed = torch.tensor(self.names["mMed"].iloc[idx]),
            mDark = torch.tensor(self.names["mDark"].iloc[idx]),
            rinv = torch.tensor(self.names["rinv"].iloc[idx]),
        )

if __name__=="__main__":
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="configs/C1_EventLevel.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    ds = args.dataset
    ft = args.features

    dataset = RootDataset(ds.path, ds.signal, ds.background, ft.eventVariables, ft.jetVariables)
    loader = udata.DataLoader(dataset=dataset, batch_size=dataset.__len__(), num_workers=0)
    x = next(iter(loader))
    print(x["data"])
    print(x["label"])
    print(x["mMed"])
    print(x["mDark"])
    print(x["rinv"])

    #sizes = get_sizes(len(dataset), ds.sample_fractions)
    #train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))
    #loader = udata.DataLoader(dataset=train, batch_size=train.__len__(), num_workers=0)
    #x = next(iter(loader))
    #print(x["data"])
    #print(x["label"])
        