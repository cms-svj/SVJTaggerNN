import numpy as np
import uproot as up
import matplotlib.pyplot as plt
import matplotlib as mpl
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import random
import torch.utils.data as udata
import torch
import math
import pandas as pd

def get_all_vars(inputFolder, samples, variables, tree="tree"):
    dSets = []
    signal = []
    for key,fileList in samples.items():
        for fileName in fileList:
            f = up.open(inputFolder  + fileName + ".root")
            branches = f[tree].pandas.df(variables)
            dSets.append(branches)
            if key == "signal":
                signal += [True]*len(branches)
            else:
                signal += [False]*len(branches)
    dataSet = pd.concat(dSets)
    return [dataSet,signal]

class RootDataset(udata.Dataset):
    def __init__(self, inputFolder, root_file, variables, signal=False):
        dataInfo = get_all_vars(inputFolder, root_file, variables)
        self.root_file = root_file
        self.variables = variables
        self.vars = dataInfo[0]
        self.signal = dataInfo[1]

    def __len__(self):
        return len(self.vars)

    def __getitem__(self, idx):
        data_np = self.vars.astype(float).values[idx]
        label_np = torch.zeros(1, dtype=torch.long)
        if self.signal[idx]:label_np += 1
        label = label_np
        data = torch.from_numpy(data_np.copy())
        return label, data

if __name__=="__main__":
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    inputFiles = []
    dSet = args.dataset
    sigFiles = dSet.signal
    inputFiles = dSet.background
    inputFiles.update(sigFiles)
    print(inputFiles)
    varSet = args.features.train
    print(varSet)
    dataset = RootDataset(dSet.path, inputFiles, varSet, signal=False)

    for i in range(10):
        print("---"*50)
        label, data = dataset.__getitem__(i)
        print(label,data)
