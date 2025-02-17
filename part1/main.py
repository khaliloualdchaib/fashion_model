import pandas as pd
from data_prep.data import clean_data, encode_csv
from torch.utils.data import random_split, DataLoader
from pipeline.FashionModel import FashionModel
from data_prep.FashionDataset import FashionDataset, transform
import torch.nn as nn
import torch.optim as optim
from pipeline.train_test import test, train, plot_training

CLASSIFICATIONS = ['style', 'pattern', 'more_attributes', 'sleeve_type', 'tops_fit', 'bottoms_fit']

if __name__ == "__main__":
    # clean_data("data/sampled_data.csv", "data/cleaned_data.csv")
    # encode_csv("data/cleaned_data.csv", "data/encoded_data.csv", CLASSIFICATIONS)

    batchsize = 64
    learning_rate = 0.0001
    num_epochs = 50

    dataset = FashionDataset(csv_file='data/encoded_data.csv', transform=transform)

    num_brand_features = len(dataset.brand_columns)
    num_classes = len(dataset.label_columns)

    model = FashionModel(num_brand_features, num_classes)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-5)

    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size 
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    val_loader  = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    # train(model, num_epochs, optimizer, loss_function, train_loader, val_loader, True, "b1")
    # plot_training(num_epochs, train_losses=losses["train_losses"], val_losses=losses["val_losses"], file_name="b1")

    
    test(model, loss_function, test_loader, dataset, CLASSIFICATIONS, "b1")