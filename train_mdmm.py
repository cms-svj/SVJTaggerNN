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
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import numpy as np
from tqdm import tqdm
from DiscoFast import distance_corr
# from Disco import distance_corr as slow_dcorr
import copy
from GPUtil import showUtilization as gpu_usage
import mdmm
import csv
from functools import partial
from dataset import RootDataset
from dataset_sampler import getDataset
# import dcor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_DEBUG"] = "INFO"

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def printProportion(args,data):
    inputFileNames = []
    ds = args.dataset
    for key, fileList in ds.background.items():
        inputFileNames += fileList
    for key, fileList in ds.signal.items():
        inputFileNames += fileList
    uval, ucounts = np.unique(data["inputFileIndices"],return_counts=True)
    print("uval, ucounts")
    for i in range(len(uval)):
        print(inputFileNames[uval[i]],ucounts[i])
        
def processBatch(args, device, data, model, criterion, lambdas):
    # printProportion(args,data)
    data, dcorrVar, label = data["data"], data["dcorrVar"], data["label"]
    l1, l2, lgr, ldc = lambdas
    #print("\n Initial GPU Usage")
    #gpu_usage()
    with autocast():
        # print("data")
        # print(data.float())
        # print("data sum")
        # print(torch.sum(data.float()))
        output = model(data.float().to(device))
        if torch.isnan(torch.sum(output)):
            print(output)
            raise Exception("output has nan")
        batch_loss = criterion(output.to(device), label.to(device)).to(device)
    #print("\n After emptying cache")
    #gpu_usage()

    # Added distance correlation calculation between tagger output and jet pT
    outSoftmax = f.softmax(output,dim=1)
    signalIndex = args.hyper.num_classes - 1
    outTag = outSoftmax[:,signalIndex]
    normedweight = torch.ones_like(outTag)
    sgpVal = dcorrVar.to(device)
    cutValue = 0
    mask = sgpVal.gt(cutValue).to(device)
    maskedoutTag = torch.masked_select(outTag, mask)
    maskedsgpVal = torch.masked_select(sgpVal, mask)
    maskedweight = torch.masked_select(normedweight, mask)
    # batch_loss_dc = distance_corr(maskedoutTag.to(device), maskedsgpVal.to(device), maskedweight.to(device), 1).to(device)
    # lambdaDC = ldc
    return l1*batch_loss, maskedoutTag.to(device), maskedsgpVal.to(device), maskedweight.to(device)

def samplerWeights(ds,hyper,trainingData,nonTrainingInfo):
    # the sampler weights are not the actual weights.
    # The purpose of the sampler weights is to make sure that each signal is represented equally in the training,
    # and each background subsample is represented proportionally. But the ratio of signal to any background = 1:1.
    uval, ucounts = np.unique(nonTrainingInfo["inputFileIndices"],return_counts=True)
    ns = len(ds.signal.keys())        # number of signal kinds
    nsf = 0
    for key, fileList in ds.signal.items():
        nsf += len(fileList)
    nb = len(ds.background.keys())    # number of background kinds
    labels = trainingData["label"].values
    trueWeights = nonTrainingInfo["w"].values
    kindWeight = 1./(ns+nb) # total weight per sample kind
    sampler_weights = np.zeros(len(labels))
    # make sure that each subsample has the same chance of getting selected regardless of how many events are in the subsample training file
    for j in range(len(uval)):
        sampler_weights[nonTrainingInfo["inputFileIndices"] == uval[j]] = 1/ucounts[j]
    # make sure the the subsample has the correct proportion
    for i in np.unique(labels):
        labCon = labels == i
        if i == hyper.num_classes - 1: # these are signals
            sampler_weights[labCon] *= kindWeight/nsf 
        else:
            bkgTrueWeight = trueWeights[labCon]
            sumOfWeights = np.sum(np.unique(bkgTrueWeight))
            sampler_weights[labCon] *= kindWeight * bkgTrueWeight / sumOfWeights
        # sampler_weights = np.ones(len(trueWeights))
    return sampler_weights

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
    train = RootDataset(trainData, trainNonTrainingInfo)
    val = RootDataset(valData, valNonTrainingInfo)
    samples_weights_tr = samplerWeights(ds,hyper,trainData,trainNonTrainingInfo)
    samples_weights_val = samplerWeights(ds,hyper,valData,valNonTrainingInfo)
    sampler_training = udata.WeightedRandomSampler( samples_weights_tr, len(samples_weights_tr),True)
    sampler_val = udata.WeightedRandomSampler( samples_weights_val, len(samples_weights_val),True)
    loader_train = udata.DataLoader(dataset=train, batch_size=hyper.batchSize, num_workers=0, shuffle=False, sampler=sampler_training)
    loader_val = udata.DataLoader(dataset=val, batch_size=hyper.batchSize, num_workers=0, shuffle=False, sampler=sampler_val)

    # Build model
    model = DNN(n_var=len(train[0]["data"]), n_layers=hyper.num_of_layers, n_nodes=hyper.num_of_nodes, n_outputs=hyper.num_classes, drop_out_p=hyper.dropout).to(device=device)
    #model = DNN_GRF(n_var=len(varSet), n_layers_features=hyper.num_of_layers_features, n_layers_tag=hyper.num_of_layers_tag, n_layers_pT=hyper.num_of_layers_pT, n_nodes=hyper.num_of_nodes, n_outputs=2, n_pTBins=hyper.n_pTBins, drop_out_p=hyper.dropout).to(device=device)
    if (args.model == None):
        #model.apply(init_weights)
        print("Creating new model ")
        args.model = 'net.pth'
    else:
        print("Loading model from " + modelLocation)
        model.load_state_dict(torch.load(modelLocation))
    modelLocation = "{}/{}".format(args.outf,args.model)
    model = copy.deepcopy(model)
    model = model.to(device)
    model.eval()
    modelInfo = []
    modelInfo.append("Model contains {} trainable parameters.".format(count_parameters(model)))
    with open('{}/modelInfo.txt'.format(args.outf), 'w') as f:
        for line in modelInfo:
            f.write("{}\n".format(line))
    
    #### mdmm ######
    fn = partial(lambda power, var_1, var_2, normedweight: distance_corr(power, var_1, var_2, normedweight), 1)
    if hyper.mdmm_type == "Equal":
        constraint = mdmm.EqConstraint(fn, hyper.mdmm_constraint, hyper.mdmm_scale, hyper.mdmm_damping)
    elif hyper.mdmm_type == "Max":
        constraint = mdmm.MaxConstraint(fn, hyper.mdmm_constraint, hyper.mdmm_scale, hyper.mdmm_damping)
    else:
        raise Exception("Please specify a valid mdmm constraint type.")
    mdmm_module = mdmm.MDMM([constraint])
    opt = mdmm_module.make_optimizer(model.parameters(), lr=hyper.learning_rate)
    writer = csv.writer(open('{}/discoLoss.csv'.format(args.outf), 'w'))
    writer.writerow(['l2_loss', 'disco_loss'])
    ################
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    criterion.to(device=device)

    ### non mdmm ####
    # optimizer = optim.Adam(model.parameters(), lr = hyper.learning_rate, weight_decay=1e-4)
    #################
    
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
        model.train()
        for i, data in tqdm(enumerate(loader_train), unit="batch", total=len(loader_train)):
            
            ### non mdmm ####
            # optimizer.zero_grad()
            #################
            batch_loss_tag, outputScore, dcorrVal, normedweight = processBatch(args, device, data, model, criterion, [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR, hyper.lambdaDC])
            batch_loss_total = batch_loss_tag # + batch_loss_dc
            
            ##### mdmm ########
            # print("Disco MET:",distance_corr(1, outputScore, dcorrVal, normedweight))
            # print("Disco logMET:",distance_corr(1, outputScore, torch.log(dcorrVal), normedweight))
            mdmm_return = mdmm_module(batch_loss_total,[[outputScore, torch.log(dcorrVal), normedweight]])
            if i % 100 == 0:
                writer.writerow([batch_loss_total.item(), *(norm.item() for norm in mdmm_return.fn_values)])
                print(f'{i} {batch_loss_total}')
                nns = outputScore.detach().cpu().numpy().astype("float")
                lmet = torch.log(dcorrVal).detach().cpu().numpy().astype("float")
                # print("Disco logMET:",slow_dcorr(1,outputScore, torch.log(dcorrVal),normedweight))
                print('Layer weight norms:',
                      *(f'{norm.item():g}' for norm in mdmm_return.fn_values))
            opt.zero_grad()
            mdmm_return.value.backward()
            opt.step()
            #####################
            
            ### non mdmm ####
            # batch_loss_total.backward()
            # optimizer.step()
            #################
            train_loss_tag += batch_loss_tag.item()
            # train_loss_dc += batch_loss_dc.item()
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

        # validation
        val_loss_tag = 0
        val_loss_dc = 0
        val_dc_val = 0
        val_loss_total = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader_val):
                output_loss_tag, outputScore, dcorrVal, normedweight = processBatch(args, device, data, model, criterion, [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR, hyper.lambdaDC])
                output_loss_total = output_loss_tag # + output_loss_dc
                val_loss_tag += output_loss_tag.item()
                # val_loss_dc += output_loss_dc.item()
                # val_dc_val += dc_val.item()
                val_loss_total += output_loss_total.item()
        val_loss_tag /= len(loader_val)
        val_loss_dc /= len(loader_val)
        #val_dc_val /= len(loader_val)
        val_loss_total /= len(loader_val)
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
        torch.save(model.state_dict(), "{}/net_{}.pth".format(args.outf,epoch))
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
    np.savez(args.outf + "/losses",training_losses_tag=training_losses_tag,validation_losses_tag=validation_losses_tag,training_losses_dc=training_losses_dc,
             validation_losses_dc=validation_losses_dc,training_losses_total=training_losses_total,validation_losses_total=validation_losses_total)

    parser.write_config(args, args.outf + "/config_out.py")

    # keep only the model with the lowest validation loss and model at the last epoch
    minInd = np.argmin(validation_losses_total)
    os.system(f"cp {args.outf}/net_{minInd}.pth {args.outf}/model.pth")
    for i in range(hyper.epochs-1):
        if i != minInd:
            os.system(f"rm {args.outf}/net_{i}.pth")
    
if __name__ == "__main__":
    main()
