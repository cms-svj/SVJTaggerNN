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
from dataset_sampler import getDataset

class RootDataset(udata.Dataset):
    def __init__(self, normedData, nonTrainingInfo):
        self.allData = normedData
        self.names = nonTrainingInfo

    def __len__(self):
        return len(self.allData)

    def __getitem__(self, idx):
        row = self.allData.iloc[idx]
        data, label = row.iloc[:-1], row.iloc[-1]
        return dict(
            data=torch.tensor(data),
            label=torch.tensor(label, dtype=torch.long),
            dcorrVar = torch.tensor(self.names["dcorrVar"].iloc[idx]),
            st = torch.tensor(self.names["st"].iloc[idx]),
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
    tr = args.training
    trainData, trainNonTrainingInfo, valData, valNonTrainingInfo, testData, testNonTrainingInfo = getDataset(ds.path, ds.signal, ds.background, ds.sample_fractions, ft.eventVariables, ft.jetVariables, ft.dcorrVar, tr.weights, ft.numOfJetsToKeep)
    dataset = RootDataset(trainData,trainNonTrainingInfo)
    loader = udata.DataLoader(dataset=dataset, batch_size=dataset.__len__(), num_workers=0)
    x = next(iter(loader))
    print("Data")
    print(x["data"])
    print("label")
    print(x["label"])
    print("mMed")
    print(x["mMed"])
    print("mDark")
    print(x["mDark"])
    print("rinv")
    print(x["rinv"])
    uniqueLabels, uniqueCounts = np.unique(x["label"],return_counts=True)
    for i in range(len(uniqueLabels)):
        uniqueLabel = uniqueLabels[i]
        uniqueCount = uniqueCounts[i]
        print(uniqueLabel,uniqueCount)