#!/bin/env python
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import torch.utils.data as udata
import torch.optim as optim
import os
from models import DNN, DNN_GRF
from dataset import RootDataset, get_sizes
import matplotlib.pyplot as plt
from magiconfig import ArgumentParser, MagiConfigOptions, ArgumentDefaultsRawHelpFormatter
from configs import configs as c
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from tqdm import tqdm
from Disco import distance_corr

# ask Kevin how to create training root files for the NN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def processBatch(args, device, data, model, criterions, lambdas, epoch):
    label, d, mcType, pTLab, pT, mT, w, med, dark, rinv, alpha = data
    l1, l2, lgr, ldc = lambdas
    with autocast():
        output, output_pTClass = model(d.float().to(device), lgr)
        criterion, criterion_pTClass = criterions
        batch_loss = criterion(output.to(device), label.squeeze(1).to(device)).to(device)
        batch_loss_pTClass = criterion_pTClass(output_pTClass.to(device), pTLab.squeeze(1).to(device)).to(device)
    pTVal = pTLab.squeeze(1)
    labVal = label.squeeze(1)

    # Added distance correlation calculation between tagger output and jet pT
    outTag = f.softmax(output,dim=1)[:,1]
    normedweight = torch.ones_like(outTag)
    # disco signal parameter
    sgpVal = dark.squeeze(1)
    mask = sgpVal.gt(0).to(device)
    maskedoutTag = torch.masked_select(outTag, mask)
    maskedsgpVal = torch.masked_select(sgpVal, mask)
    maskedweight = torch.masked_select(normedweight, mask)
    batch_loss_dc = distance_corr(maskedoutTag.to(device), maskedsgpVal.to(device), maskedweight.to(device), 1).to(device)
    lambdaDC = ldc
    return l1*batch_loss, l2*batch_loss_pTClass, lambdaDC*batch_loss_dc, batch_loss_dc

def main():
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
    dSet = args.dataset
    sigFiles = dSet.signal
    inputFiles = dSet.background
    hyper = args.hyper
    inputFiles.update(sigFiles)
    print(inputFiles)
    varSet = args.features.train
    print(varSet)
    pTBins = hyper.pTBins
    uniform = args.features.uniform
    mT = args.features.mT
    weight = args.features.weight
    dataset = RootDataset(inputFolder=dSet.path, root_file=inputFiles, variables=varSet, pTBins=pTBins, uniform=uniform, mT=mT, weight=weight)
    sizes = get_sizes(len(dataset), dSet.sample_fractions)
    train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))
    loader_train = udata.DataLoader(dataset=train, batch_size=hyper.batchSize, num_workers=4, shuffle=True)
    loader_val = udata.DataLoader(dataset=val, batch_size=hyper.batchSize, num_workers=4)
    loader_test = udata.DataLoader(dataset=test, batch_size=hyper.batchSize, num_workers=4)
    # Build model
    #model = DNN(n_var=len(varSet), n_layers=hyper.num_of_layers, n_nodes=hyper.num_of_nodes, n_outputs=2, drop_out_p=hyper.dropout).to(device=device)
    model = DNN_GRF(n_var=len(varSet), n_layers_features=hyper.num_of_layers_features, n_layers_tag=hyper.num_of_layers_tag, n_layers_pT=hyper.num_of_layers_pT, n_nodes=hyper.num_of_nodes, n_outputs=2, n_pTBins=hyper.n_pTBins, drop_out_p=hyper.dropout).to(device=device)
    if (args.model == None):
        #model.apply(init_weights)
        print("Creating new model ")
        args.model = 'net.pth'
    else:
        print("Loading model from " + modelLocation)
        model.load_state_dict(torch.load(modelLocation))
        model.eval()
    modelLocation = "{}/{}".format(args.outf,args.model)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    criterion_pTClass = nn.CrossEntropyLoss()
    criterion.to(device=device)
    criterion_pTClass.to(device=device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = hyper.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95, last_epoch=-1, verbose=True)

    # training and validation
    writer = SummaryWriter()
    training_losses_tag = np.zeros(hyper.epochs)
    training_losses_pTClass = np.zeros(hyper.epochs)
    training_losses_dc = np.zeros(hyper.epochs)
    training_losses_total = np.zeros(hyper.epochs)
    validation_losses_tag = np.zeros(hyper.epochs)
    validation_losses_pTClass = np.zeros(hyper.epochs)
    validation_losses_dc = np.zeros(hyper.epochs)
    validation_losses_total = np.zeros(hyper.epochs)
    for epoch in range(hyper.epochs):
        print("Beginning epoch " + str(epoch))
        # training
        train_loss_tag = 0
        train_loss_pTClass = 0
        train_loss_dc = 0
        train_dc_val = 0
        train_loss_total = 0
        for i, data in tqdm(enumerate(loader_train), unit="batch", total=len(loader_train)):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_loss_tag, batch_loss_pTClass, batch_loss_dc, dc_val = processBatch(args, device, data, model, [criterion, criterion_pTClass], [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR, hyper.lambdaDC], epoch)
            batch_loss_total = batch_loss_tag + batch_loss_pTClass + batch_loss_dc
            batch_loss_total.backward()
            optimizer.step()
            model.eval()
            train_loss_tag += batch_loss_tag.item()
            train_loss_pTClass += batch_loss_pTClass.item()
            train_loss_dc += batch_loss_dc.item()
            train_dc_val += dc_val.item()
            train_loss_total+=batch_loss_total.item()
            writer.add_scalar('training loss', train_loss_total / 1000, epoch * len(loader_train) + i)
        train_loss_tag /= len(loader_train)
        train_loss_pTClass /= len(loader_train)
        train_loss_dc /= len(loader_train)
        train_dc_val /= len(loader_train)
        train_loss_total /= len(loader_train)
        training_losses_tag[epoch] = train_loss_tag
        training_losses_pTClass[epoch] = train_loss_pTClass
        training_losses_dc[epoch] = train_loss_dc
        training_losses_total[epoch] = train_loss_total
        print("t_tag: "+ str(train_loss_tag))
        print("t_pTClass: "+ str(train_loss_pTClass))
        print("t_dc: "+ str(train_loss_dc))
        print("t_dc_val: "+ str(train_dc_val))
        print("t_total: "+ str(train_loss_total))

        # validation
        val_loss_tag = 0
        val_loss_pTClass = 0
        val_loss_dc = 0
        val_dc_val = 0
        val_loss_total = 0
        for i, data in enumerate(loader_val):
            output_loss_tag, output_loss_pTClass, output_loss_dc, dc_val = processBatch(args, device, data, model, [criterion, criterion_pTClass], [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR, hyper.lambdaDC], epoch)
            output_loss_total = output_loss_tag + output_loss_pTClass + output_loss_dc
            val_loss_tag += output_loss_tag.item()
            val_loss_pTClass += output_loss_pTClass.item()
            val_loss_dc += output_loss_dc.item()
            val_dc_val += dc_val.item()
            val_loss_total += output_loss_total.item()
        val_loss_tag /= len(loader_val)
        val_loss_pTClass /= len(loader_val)
        val_loss_dc /= len(loader_val)
        val_dc_val /= len(loader_val)
        val_loss_total /= len(loader_val)
        scheduler.step()
        #scheduler.step(torch.tensor([val_loss_total]))
        validation_losses_tag[epoch] = val_loss_tag
        validation_losses_pTClass[epoch] = val_loss_pTClass
        validation_losses_dc[epoch] = val_loss_dc
        validation_losses_total[epoch] = val_loss_total
        print("v_tag: "+ str(val_loss_tag))
        print("v_pTClass: "+ str(val_loss_pTClass))
        print("v_dc: "+ str(val_loss_dc))
        print("v_dc_val: "+ str(val_dc_val))
        print("v_total: "+ str(val_loss_total))
        # save the model
        model.eval()
        torch.save(model.state_dict(), modelLocation)
    writer.close()

    # plot loss/epoch for training and validation sets
    print("Making basic validation plots")
    training_tag = plt.plot(training_losses_tag, label='training_tag')
    validation_tag = plt.plot(validation_losses_tag, label='validation_tag')
    training_pTClass = plt.plot(training_losses_pTClass, label='training_pTClass')
    validation_pTClass = plt.plot(validation_losses_pTClass, label='validation_pTClass')
    training_dc = plt.plot(training_losses_dc, label='training_dc')
    validation_dc = plt.plot(validation_losses_dc, label='validation_dc')
    training_total = plt.plot(training_losses_total, label='training_total')
    validation_total = plt.plot(validation_losses_total, label='validation_total')
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(args.outf + "/loss_plot.png")
    np.savez(args.outf + "/normMeanStd",normMean=dataset.normMean,normStd=dataset.normstd)
    parser.write_config(args, args.outf + "/config_out.py")

if __name__ == "__main__":
    main()
