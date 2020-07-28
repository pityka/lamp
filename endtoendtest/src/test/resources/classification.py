from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import itertools
import torch.utils.data as data
from os.path import isfile

class CsvDataset(data.Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.X = self.df.drop("target", axis=1)
        self.y = self.df["target"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]

class Net(nn.Module):

    def __init__(self, D_in, D_out):
        super().__init__()
        h = 4
        self.fc1 = nn.Linear(D_in, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, D_out)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.gelu3 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(h)
        self.bn2 = nn.BatchNorm1d(h)
        self.bn3 = nn.BatchNorm1d(D_out)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.gelu2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.gelu3(x)

        x = self.lsm(x)

        return x.squeeze()


    
def train(csv_file):
    """
    """
    n_epochs = 50
    dataset = CsvDataset(csv_file)

    # Split into training and test
    split = int(len(dataset)/3.0)
    testset = range(0,split+1)
    trainset = data.Subset(dataset,range(split+1,len(dataset)))

    # Dataloaders
    trainloader = data.DataLoader(trainset, batch_size=1024, shuffle=True, drop_last = False)

    # Use gpu if available
    device = "cpu"

    # Define the model
    numclasses = int(dataset.y.max()+1)
    numfeatures = dataset.X.shape[1]
    
    net = Net(numfeatures, numclasses).to(device)

    # Loss function
    lossFunction = nn.NLLLoss()

    # Optimizer
    optimizer = optim.AdamW(net.parameters(), weight_decay=0.0001, lr=0.001)

    # Train the net
    for epoch in range(n_epochs):

        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            output = net(inputs.float())
            loss = lossFunction(output, labels.long())
            loss.backward()
            optimizer.step()

    # Comparing training to test
    testFeatures = torch.Tensor(dataset.X.iloc[testset].values).to(device)
    testOutput = net(testFeatures.float()).detach().cpu().numpy()
    testPredicted = np.argmax(testOutput,axis=1)
    testTruthLabels = dataset.y.iloc[testset]
    accuracy = testTruthLabels.eq(testPredicted).mean()
    return accuracy

if __name__ == "__main__":
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    args = parser.parse_args()

    # Call the main function of the script
    accuracy = train(args.file)
    print(accuracy)