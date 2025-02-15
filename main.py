import pandas as pd
from data import clean_data, encode_csv, resize_with_padding
from torch.utils.data import random_split, DataLoader
from FashionModel import FashionModel
from FashionDataset import FashionDataset, transform
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

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

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, brands, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images, brands)
            loss = loss_function(outputs, labels)                     
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, brands, labels in val_loader:
                outputs = model(images, brands)
                val_loss += loss_function(outputs, labels).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
    
    file_name = "b0_xtra_transformation_dropout_50"
    # Plot training and validation loss
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(f'plots/{file_name}.png')

    torch.save(model.state_dict(), f'models/{file_name}.pth')

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, brands, labels in test_loader:
            outputs = model(images, brands)
            test_loss += loss_function(outputs, labels).item()
            predicted_probabilities = torch.sigmoid(outputs)
            predicted = (predicted_probabilities > 0.5).float()
            correct += (predicted == labels).all(dim=1).sum().item()
            total += labels.size(0)
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
