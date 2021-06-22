import torch
import torch.nn as nn
from torch.nn import functional as f
import torch.utils.data as udata
import os
from models import DNN, DNN_GRF
from dataset import RootDataset, get_sizes
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score

# ask Kevin how to create training root files for the NN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mplColors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def getNNOutput(dataset, model):
    loader = udata.DataLoader(dataset=dataset, batch_size=dataset.__len__(), num_workers=0)
    l, d, p, m, w = next(iter(loader))
    labels = l.squeeze(1).numpy()
    data = d.float()
    pT = p.squeeze(1).float().numpy()
    mT = m.squeeze(1).float().numpy()
    weight = w.squeeze(1).float().numpy()
    model.eval()
    out, _ = model(data)
    output = f.softmax(out,dim=1)[:,1].detach().numpy()
    return labels, output, pT, mT, weight

def getROCStuff(label, output):
    fpr, tpr, thresholds = roc_curve(label, output)
    auc = roc_auc_score(label, output)
    return fpr, tpr, auc

def getSgBgOutputs(label, output):
    y_Sg = []
    y_Bg = []
    for lt in range(len(label)):
        lbl = label[lt]
        if lbl == 1:
            y_Sg.append(output[lt])
        else:
            y_Bg.append(output[lt])
    return y_Sg, y_Bg

def histplot(data,bins,color,label,weights=None,alpha=1.0,hatch=None,points=False,facecolorOn=True):
    data,bins = np.histogram(data, bins=bins, weights=weights, density=True)
    bins = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
    binwidth = bins[1] - bins[0]
    pbins = np.append(bins,bins[-1]+binwidth)
    pdata = np.append(data,data[-1])
    if points:
        pbins = pbins + binwidth/2.
        plt.plot(pbins,pdata,color=color,label=label,marker=".",linestyle="None")
    else:
        plt.step(pbins,pdata,where="post",color=color)
        if facecolorOn:
            facecolor=color
        else:
            facecolor="none"
        plt.fill_between(pbins,pdata, step="post", edgecolor=color, facecolor=facecolor, label=label, alpha=alpha, hatch=hatch)

def plotByBin(binVar,binVarBins,histVar,xlabel,varLab,outDir,plotName):
    binWidth = binVarBins[1] - binVarBins[0]
    fig, ax = plt.subplots(figsize=(12, 8))
    for j in range(len(binVarBins)):
        binVal = binVarBins[j]
        if j < len(binVarBins)-1:
            histVL = histVar[np.absolute(binVal-binVar) < binWidth/2.]
            lab = '{:.2f} < {} < {:.2f}'.format(binVarBins[j]-binWidth/2.,varLab,binVarBins[j]+binWidth/2)
        else:
            histVL = histVar[binVar > binVal - binWidth/2. ]
            lab = '{} > {:.2f}'.format(varLab,binVarBins[j]-binWidth/2.)
        histplot(histVL,bins=50,color=mplColors[j],alpha=0.5,label=lab)
    ax.set_ylabel('Norm Events')
    ax.set_xlabel(xlabel)
    plt.yscale("log")
    plt.legend()
    plt.savefig(outDir + "/{}.pdf".format(plotName), dpi=fig.dpi)

def main():
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="configs/C1.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("--outf", type=str, default="logs", help='Name of folder to be used to store outputs')
    parser.add_argument("--model", type=str, default=None, help="Existing model to continue training, if applicable")
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()
    parser.write_config(args, args.outf + "/config_out.py")

    # Choose cpu or gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', args.device)
    if args.device.type == 'cuda':
        gpuIndex = torch.cuda.current_device()
        print("Using GPU named: \"{}\"".format(torch.cuda.get_device_name(gpuIndex)))

    # Load dataset
    print('Loading dataset ...')
    dSet = args.dataset
    sigFiles = dSet.signal
    inputFiles = dSet.background
    hyper = args.hyper
    inputFiles.update(sigFiles)
    varSet = args.features.train
    uniform = args.features.uniform
    mT = args.features.mT
    weight = args.features.weight
    dataset = RootDataset(inputFolder=dSet.path, root_file=inputFiles, variables=varSet, uniform=uniform, mT=mT, weight=weight)
    sizes = get_sizes(len(dataset), dSet.sample_fractions)
    train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))
    loader_train = udata.DataLoader(dataset=train, batch_size=hyper.batchSize, num_workers=0)
    loader_val = udata.DataLoader(dataset=val, batch_size=hyper.batchSize, num_workers=0)
    loader_test = udata.DataLoader(dataset=test, batch_size=hyper.batchSize, num_workers=0)
    df = dataset.dataFrame
    # # Build model
    model = DNN_GRF(n_var=len(varSet), n_layers=hyper.num_of_layers, n_nodes=hyper.num_of_nodes, n_outputs=2, drop_out_p=hyper.dropout).to(device=args.device)
    print("Loading model from file " + args.model)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.to('cpu')
    label_train, output_train, pT_train, mT_train, w_train = getNNOutput(train, model)
    label_test, output_test, pT_test, mT_test, w_test = getNNOutput(test, model)
    fpr_Train, tpr_Train, auc_Train = getROCStuff(label_train, output_train)
    fpr_Test, tpr_Test, auc_Test = getROCStuff(label_test, output_test)

    # plot correlation
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = np.round(df.corr(),2)
    ax = sns.heatmap(corr,cmap="Spectral",annot=True)
    bottom, top = ax.get_ylim()
    left, right = ax.get_xlim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xlim(left - 0.5, right + 0.5)
    plt.savefig(args.outf + "/corrHeatMap.pdf", dpi=fig.dpi)
    fig, ax = plt.subplots(figsize=(12, 8))

    # plot input variable
    print(varSet)
    df["label"] = np.array(dataset.signal)[:,0]
    print(df[:10])
    # weights = np.array(df["procweight"]*df["puweight"])
    for var in varSet:
        dataSig = df[var][df["label"] == 1]
        dataBkg = df[var][df["label"] == 0]
        fig, ax = plt.subplots(figsize=(12, 8))
        histplot(dataSig,bins=50,color='xkcd:blue',alpha=0.5,label='Background')
        histplot(dataBkg,bins=50,color='xkcd:red',alpha=1.0,label='Signal',hatch="//",facecolorOn=False)
        ax.set_ylabel('Norm Events')
        ax.set_xlabel(var)
        plt.legend()
        plt.savefig(args.outf + "/{}.pdf".format(var), dpi=fig.dpi)

    # plot ROC curve
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve', pad=45.0)
    plt.plot(fpr_Train, tpr_Train, label="Train (area = {:.3f})".format(auc_Train), color='xkcd:red')
    plt.plot(fpr_Test, tpr_Test, label="Test (area = {:.3f})".format(auc_Test), color='xkcd:black')
    plt.legend(loc='best')
    fig.savefig(args.outf + "/roc_plot.pdf", dpi=fig.dpi)
    plt.close(fig)

    # plot eff vs pT
    binWidth = 500
    wpt = 0.5 # NN working point
    pTX = np.arange(250,3000,binWidth)
    totSigJet = len(label_train[label_train==1])
    totBkgJet = len(label_train) - totSigJet
    eff = []
    for j in range(len(pTX)):
        binVal = pTX[j]
        if j < len(pTX)-1:
            pTCond = np.absolute(binVal-pT_train) < binWidth/2.
        else:
            pTCond = pT_train > binVal - binWidth/2.
        output_pTC = output_train[pTCond]
        label_pTC = label_train[pTCond]
        output_pTC_wpt = np.where(output_pTC>wpt,1,0)
        match_output_label = np.where(label_pTC == output_pTC,1.0,0.0)
        eff.append(np.sum(match_output_label)/totSigJet)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(pTX,eff)
    ax.set_ylabel('Efficiency')
    ax.set_xlabel('pT (GeV)')
    plt.savefig(args.outf + "/effVspT.pdf", dpi=fig.dpi)

    # # histogram NN score
    fig, ax = plt.subplots(figsize=(12, 8))
    histplot(output_train,bins=50,color='xkcd:blue',alpha=0.5,label='Training set')
    ax.set_ylabel('Norm Events')
    ax.set_xlabel("NN Score")
    plt.legend()
    plt.savefig(args.outf + "/SNN.pdf", dpi=fig.dpi)

    # NN score per pT bin
    plotByBin(binVar=pT_train,binVarBins = np.arange(250,3000,500),histVar=output_train,xlabel="NN Score",varLab="pT",outDir=args.outf,plotName="SNNperpT")
    # pT per NN score bin
    plotByBin(binVar=output_train,binVarBins = np.arange(0,1,0.1),histVar=pT_train,xlabel="$Jet p_T (GeV)$",varLab="SNN",outDir=args.outf,plotName="pTperSNN")

    # plot discriminator
    bins = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    y_Train_Sg, y_Train_Bg = getSgBgOutputs(label_train, output_train)
    y_test_Sg, y_test_Bg = getSgBgOutputs(label_test, output_test)
    ax.set_title('')
    ax.set_ylabel('Norm Events')
    ax.set_xlabel('Discriminator')
    histplot(y_test_Sg,bins,color='xkcd:red',alpha=1.0,label='Sg Test',hatch="//",facecolorOn=False)
    histplot(y_test_Bg,bins,color='xkcd:blue',alpha=0.5,label='Bg Test')
    histplot(y_Train_Sg, bins, color='xkcd:red',label='Sg Train',points=True)
    histplot(y_Train_Bg, bins, color='xkcd:blue',label='Bg Train',points=True)
    ax.legend(loc='best', frameon=False)
    fig.savefig(args.outf + "/discriminator.pdf", dpi=fig.dpi)

if __name__ == "__main__":
    main()