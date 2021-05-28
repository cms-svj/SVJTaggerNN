#!/bin/env python
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as f
from torch.utils.tensorboard import SummaryWriter
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

# ask Kevin how to create training root files for the NN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def getNNOutput(dataset, model):
    loader = udata.DataLoader(dataset=dataset, batch_size=dataset.__len__(), num_workers=0)
    l, d, p = next(iter(loader))
    labels = l.squeeze(1).numpy()
    data = d.float()
    pT = p.squeeze(1).float().numpy()
    model.eval()
    out, _ = model(data)
    output = f.softmax(out,dim=1)[:,1].detach().numpy()
    return labels, output, pT

def processBatch(args, data, model, criterions, lambdas):
    label, d, pt = data
    l1, l2, lgr = lambdas
    output, output_reg = model(d.float().to(args.device), lgr)
    criterion, criterion_reg = criterions
    batch_loss = criterion(output.to(args.device), label.squeeze(1).to(args.device)).to(args.device)
    batch_loss_reg = criterion_reg(output_reg.to(args.device), pt.to(args.device)).to(args.device)
    return l1*batch_loss + l2*batch_loss_reg

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
        #print('Memory Usage:')
        #print('\tAllocated:', round(torch.cuda.memory_allocated(gpuIndex)/1024**3,1), 'GB')
        #print('\tCached:   ', round(torch.cuda.memory_reserved(gpuIndex)/1024**3,1), 'GB')

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
    uniform = args.features.uniform
    dataset = RootDataset(inputFolder=dSet.path, root_file=inputFiles, variables=varSet, uniform=uniform)
    sizes = get_sizes(len(dataset), dSet.sample_fractions)
    train, val, test = udata.random_split(dataset, sizes, generator=torch.Generator().manual_seed(42))
    loader_train = udata.DataLoader(dataset=train, batch_size=hyper.batchSize, num_workers=0)
    loader_val = udata.DataLoader(dataset=val, batch_size=hyper.batchSize, num_workers=0)
    loader_test = udata.DataLoader(dataset=test, batch_size=hyper.batchSize, num_workers=0)

    # Build model
    #model = DNN(n_var=len(varSet), n_layers=hyper.num_of_layers, n_nodes=hyper.num_of_nodes, n_outputs=2, drop_out_p=hyper.dropout).to(device=args.device)
    model = DNN_GRF(n_var=len(varSet), n_layers=hyper.num_of_layers, n_nodes=hyper.num_of_nodes, n_outputs=2, drop_out_p=hyper.dropout).to(device=args.device)
    if (args.model == None):
        model.apply(init_weights)
        print("Creating new model ")
    else:
        print("Loading model from file " + args.model)
        model.load_state_dict(torch.load(args.model))
        model.eval()

    # Loss function
    criterion = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    criterion.to(device=args.device)
    criterion_reg.to(device=args.device)

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr = hyper.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=10, verbose=True)

    # training and validation
    writer = SummaryWriter()
    training_losses = np.zeros(hyper.epochs)
    validation_losses = np.zeros(hyper.epochs)
    for epoch in range(hyper.epochs):
        print("Beginning epoch " + str(epoch))
        # training
        train_loss = 0
        for i, data in tqdm(enumerate(loader_train), unit="batch", total=len(loader_train)):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_loss_total = processBatch(args, data, model, [criterion, criterion_reg], [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR])
            batch_loss_total.backward()
            optimizer.step()
            model.eval()
            train_loss+=batch_loss_total.item()
            writer.add_scalar('training loss', train_loss / 1000, epoch * len(loader_train) + i)
        train_loss /= len(loader_train)
        training_losses[epoch] = train_loss
        print("t: "+ str(train_loss))

        # validation
        val_loss = 0
        for i, data in enumerate(loader_val):
            output_loss_total = processBatch(args, data, model, [criterion, criterion_reg], [hyper.lambdaTag, hyper.lambdaReg, hyper.lambdaGR])
            val_loss+=output_loss_total.item()
        val_loss /= len(loader_val)
        scheduler.step(torch.tensor([val_loss]))
        validation_losses[epoch] = val_loss
        print("v: "+ str(val_loss))

        # save the model
        model.eval()
        torch.save(model.state_dict(), os.path.join(args.outf, 'net.pth'))
    writer.close()

    # plot loss/epoch for training and validation sets
    print("Making validation plots")
    training = plt.plot(training_losses, label='training')
    validation = plt.plot(validation_losses, label='validation')
    plt.legend()
    plt.savefig(args.outf + "/loss_plot.png")

if __name__ == "__main__":
    main()
