#!/bin/env python
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
# from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import torch.utils.data as udata
import torch.optim as optim
import os
from models import DNN, DNN_GRF
from dataset import RootDataset
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import numpy as np
from tqdm import tqdm
from Disco import distance_corr
import copy
from GPUtil import showUtilization as gpu_usage
from dataset_sampler import getDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_DEBUG"] = "INFO"

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def processBatch(args, device, data, model, criterion, lambdas):
    data, dcorrVar, label = data["data"], data["dcorrVar"], data["label"]
    l1, l2, lgr, ldc = lambdas
    #print("\n Initial GPU Usage")
    #gpu_usage()
    with autocast():
        output = model(data.float().to(device))
        if torch.isnan(torch.sum(output)):
            print("output has nan:", output)
        batch_loss = criterion(output.to(device), label.to(device)).to(device)
    torch.cuda.empty_cache()
    #print("\n After emptying cache")
    #gpu_usage()

    # Added distance correlation calculation between tagger output and jet pT
    outSoftmax = f.softmax(output,dim=1)
    signalIndex = args.hyper.num_classes - 1
    outTag = outSoftmax[:,signalIndex]
    normedweight = torch.ones_like(outTag)
    # disco signal parameter
    sgpVal = dcorrVar.to(device)
    mask = sgpVal.gt(signalIndex-1).to(device)
    maskedoutTag = torch.masked_select(outTag, mask)
    maskedsgpVal = torch.masked_select(sgpVal, mask)
    maskedweight = torch.masked_select(normedweight, mask)
    batch_loss_dc = distance_corr(1,maskedoutTag.to(device), maskedsgpVal.to(device), maskedweight.to(device)).to(device)
    lambdaDC = ldc
    return l1*batch_loss, lambdaDC*batch_loss_dc

def plotEverything(trainData, trainNonTrainingInfo, kind="train"):
# ['njetsAK8', 'mT', 'METrHT_pt30', 'dEtaj12AK8', 'dRJ12AK8',
       # 'dPhiMinjMETAK8', 'jPtAK8[0]', 'jPtAK8[1]', 'jEtaAK8[0]', 'jEtaAK8[1]',
       # 'jPhiAK8[0]', 'jPhiAK8[1]', 'jEAK8[0]', 'jEAK8[1]', 'dPhijMETAK8[0]',
       # 'dPhijMETAK8[1]', 'label']
    labels = trainData["label"]
    labDict = {}
    labDict["qcd"] = labels == 0
    labDict["ttj"] = labels == 1
    labDict["svj"] = labels == 2
    for inVar in list(trainData.columns):
        varData = trainData[inVar]
        plt.figure()
        if inVar == "label":
            h,b,d = plt.hist(varData[labDict["qcd"]],bins=np.arange(0,4,1),label="qcd",alpha=0.3)
        else:
            h,b,d = plt.hist(varData[labDict["qcd"]],bins=50,label="qcd",alpha=0.3)
        for label,labCond in labDict.items():
            if label == "qcd": continue
            plt.hist(varData[labCond],bins=b,label=label,alpha=0.3)
        plt.legend()
        plt.xlabel(inVar)
        plt.savefig(f"{inVar}_{kind}.png")
    plt.figure()
    varData = trainNonTrainingInfo["dcorrVar"]
    h,b,d = plt.hist(varData[labDict["qcd"]],bins=50,label="qcd",alpha=0.3)
    for label,labCond in labDict.items():
        if label == "qcd": continue
        plt.hist(varData[labCond],bins=b,label=label,alpha=0.3)
    plt.legend()
    plt.xlabel("met")
    plt.savefig(f"dcorrVar_{kind}.png")

def main():
    rng = np.random.RandomState(2022)
    # parse arguments
    parser = ArgumentParser(config_options=MagiConfigOptions(strict = True, default="configs/C1.py"),formatter_class=ArgumentDefaultsRawHelpFormatter)
    parser.add_argument("--outf", type=str, default="logs", help='Name of folder to be used to store outputs')
    parser.add_argument("--model", type=str, default=None, help="Existing model to continue training, if applicable")
    parser.add_config_only(*c.config_schema)
    parser.add_config_only(**c.config_defaults)
    args = parser.parse_args()

    if not os.path.isdir(args.outf):
        os.mkdir(args.outf)

    # Choose cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        gpuIndex = torch.cuda.current_device()
        print("Using GPU named: \"{}\"".format(torch.cuda.get_device_name(gpuIndex)))
        #print('Memory Usage:')
        #print('\tAllocated:', round(torch.cuda.memory_allocated(gpuIndex)/1024**3,1), 'GB')
        #print('\tCached:   ', round(torch.cuda.memory_reserved(gpuIndex)/1024**3,1), 'GB')
    torch.manual_seed(args.hyper.rseed)

    # Load dataset
    print('Loading dataset ...')
    ds = args.dataset
    hyper = args.hyper
    ft = args.features
    tr = args.training
    trainData, trainNonTrainingInfo, trainData_nonUniform, trainNonTrainingInfo_nonUniform, valData, valNonTrainingInfo, testData, testNonTrainingInfo = getDataset(ds.path, ds.signal, ds.background, ds.sample_fractions, ft.eventVariables, ft.jetVariables, ft.dcorrVar, tr.weights, ft.numOfJetsToKeep,ds.flatMET)
    # sanity check: plotting input variables
    # plotEverything(trainData, trainNonTrainingInfo, kind="train")
    # plotEverything(valData, valNonTrainingInfo, kind="val")
    train = RootDataset(trainData, trainNonTrainingInfo)
    val = RootDataset(valData, valNonTrainingInfo)
    loader_train = udata.DataLoader(dataset=train, batch_size=hyper.batchSize, num_workers=0, shuffle=True)
    loader_val = udata.DataLoader(dataset=val, batch_size=hyper.batchSize, num_workers=0, shuffle=False)

    # Build model
    model = DNN(n_var=len(train[0]["data"]), n_layers=hyper.num_of_layers, n_nodes=hyper.num_of_nodes, n_outputs=hyper.num_classes, drop_out_p=hyper.dropout).to(device=device)
    if (args.model == None):
        #model.apply(init_weights)
        print("Creating new model ")
        args.model = 'net.pth'
    else:
        print("Loading model from " + modelLocation)
        model.load_state_dict(torch.load(modelLocation))
        model.eval()
    modelLocation = "{}/{}".format(args.outf,args.model)
    model = copy.deepcopy(model)
    model = model.to(device)
    model.eval()
    modelInfo = []
    modelInfo.append("Model contains {} trainable parameters.".format(count_parameters(model)))
    with open('{}/modelInfo.txt'.format(args.outf), 'w') as f:
        for line in modelInfo:
            f.write("{}\n".format(line))

    # Loss function
    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = hyper.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95, last_epoch=-1, verbose=True)

    # training and validation
    # writer = SummaryWriter()
    training_losses_tag = np.zeros(hyper.epochs)
    training_losses_dc = np.zeros(hyper.epochs)
    training_losses_total = np.zeros(hyper.epochs)
    validation_losses_tag = np.zeros(hyper.epochs)
    validation_losses_dc = np.zeros(hyper.epochs)
    validation_losses_total = np.zeros(hyper.epochs)
    for epoch in range(hyper.epochs):
        print("Beginning epoch " + str(epoch))
        # training
        train_loss_tag = 0
        train_loss_dc = 0
        train_dc_val = 0
        train_loss_total = 0
        for i, data in tqdm(enumerate(loader_train), unit="batch", total=len(loader_train)):
            model.train()
            optimizer.zero_grad()
            batch_loss_tag, batch_loss_dc = processBatch(args, device, data, model, criterion, [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR, hyper.lambdaDC])
            batch_loss_total = batch_loss_tag + batch_loss_dc
            batch_loss_total.backward()
            optimizer.step()
            train_loss_tag += batch_loss_tag.item()
            train_loss_dc += batch_loss_dc.item()
            #train_dc_val += dc_val.item()
            train_loss_total += batch_loss_total.item()
            # writer.add_scalar('training loss', train_loss_total / 1000, epoch * len(loader_train) + i)
        train_loss_tag /= len(loader_train)
        train_loss_dc /= len(loader_train)
        #train_dc_val /= len(loader_train)
        train_loss_total /= len(loader_train)
        training_losses_tag[epoch] = train_loss_tag
        training_losses_dc[epoch] = train_loss_dc
        training_losses_total[epoch] = train_loss_total
        if np.isnan(train_loss_tag):
            print("nan in training")
            break
        print("t_tag: "+ str(train_loss_tag))
        print("t_dc: "+ str(train_loss_dc))
        #print("t_dc_val: "+ str(train_dc_val))
        print("t_total: "+ str(train_loss_total))
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        # validation
        # Disable gradient computation and reduce memory consumption.
        val_loss_tag = 0
        val_loss_dc = 0
        val_dc_val = 0
        val_loss_total = 0
        with torch.no_grad():
            for i, data in enumerate(loader_val):
                output_loss_tag, output_loss_dc = processBatch(args, device, data, model, criterion, [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR, hyper.lambdaDC])
                output_loss_total = output_loss_tag + output_loss_dc
                val_loss_tag += output_loss_tag.item()
                val_loss_dc += output_loss_dc.item()
                # val_dc_val += dc_val.item()
                val_loss_total += output_loss_total.item()
        val_loss_tag /= len(loader_val)
        val_loss_dc /= len(loader_val)
        #val_dc_val /= len(loader_val)
        val_loss_total /= len(loader_val)
        # scheduler.step()
        #scheduler.step(torch.tensor([val_loss_total]))
        validation_losses_tag[epoch] = val_loss_tag
        validation_losses_dc[epoch] = val_loss_dc
        validation_losses_total[epoch] = val_loss_total
        if np.isnan(val_loss_tag):
            print("nan in val")
            break
        print("v_tag: "+ str(val_loss_tag))
        print("v_dc: "+ str(val_loss_dc))
        #print("v_dc_val: "+ str(val_dc_val))
        print("v_total: "+ str(val_loss_total))
        # save the model
        model.eval()
        # torch.save(model.state_dict(), "{}/net_{}.pth".format(args.outf,epoch))
        torch.cuda.empty_cache()
        np.savez(args.outf + "/losses",training_losses_tag=training_losses_tag,validation_losses_tag=validation_losses_tag,training_losses_dc=training_losses_dc,
             validation_losses_dc=validation_losses_dc,training_losses_total=training_losses_total,validation_losses_total=validation_losses_total)
    # save the model
    torch.save(model.state_dict(), modelLocation)
    # writer.close()

    # plot loss/epoch for training and validation sets
    print("Making loss plot")
    fig, ax = plt.subplots()
    ax.plot(training_losses_tag, label='training_tag')
    ax.plot(validation_losses_tag, label='validation_tag')
    ax.plot(training_losses_total, label='training_total')
    ax.plot(validation_losses_total, label='validation_total')
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax2 = ax.twinx()
    ax2.plot(training_losses_dc, label='training_dc',linestyle="--")
    ax2.plot(validation_losses_dc, label='validation_dc',linestyle="--")
    ax2.set_ylabel("Disco Loss")
    ax.legend(loc="upper right")
    ax2.legend(loc="center right")
    plt.savefig(args.outf + "/loss_plot.png")

    parser.write_config(args, args.outf + "/config_out.py")

if __name__ == "__main__":
    main()
