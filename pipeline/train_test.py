import torch
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

def train(model, num_epochs, optimizer, loss_function, train_loader, val_loader, save=True, filename="model"):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, brands, labels,_ in train_loader:
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
            for images, brands, labels,_ in val_loader:
                outputs = model(images, brands)
                val_loss += loss_function(outputs, labels).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
    if save:
        torch.save(model.state_dict(), f'models/{filename}.pth')
    return {"train_losses": train_losses, "val_losses": val_losses}

def plot_training(num_epochs, train_losses, val_losses, file_name="model"):
    # Plot training and validation loss
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig(f'plots/{file_name}.png')



def results2csv(barcodes, all_predictions, dataset, columns):
    label_columns = dataset.label_columns
    predictions_dict = {column: [] for column in label_columns}
    for prediction in all_predictions:
        for i, value in enumerate(prediction):
            predictions_dict[label_columns[i]].append(value)
    df_dict = {col: [] for col in columns}
    for i in range(len(predictions_dict[label_columns[0]])):
        for col1 in df_dict:
            item = []
            for col2 in predictions_dict:
                if col2.startswith(col1):
                    if predictions_dict[col2][i] == 1:
                        rest_of_string = col2[len(col1)+1:]
                        item.append(rest_of_string)
            if len(item) == 0:
                df_dict[col1].append(None)
            else:
                df_dict[col1].append(";".join(item))
    df = pd.DataFrame(df_dict)
    df["barcode"] =  [str(b.item()) if isinstance(b, torch.Tensor) else str(b) for b in barcodes]
    df.to_csv("output.csv", index=False)  

def test(model, loss_function, test_loader, dataset, columns, model_file="model"):
    model.load_state_dict(torch.load(f"models/{model_file}.pth"))
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_barcodes = []  # To store barcodes

    with torch.no_grad():
        for images, brands, labels,barcodes in tqdm(test_loader, desc="Testing"):
            outputs = model(images, brands)
            test_loss += loss_function(outputs, labels).item()

            predicted_probabilities = torch.sigmoid(outputs)
            predicted = (predicted_probabilities > 0.5).float()

            correct += (predicted == labels).all(dim=1).sum().item()
            total += labels.size(0)

            # Store labels and predictions for Hamming Loss
            all_labels.append(labels.cpu())
            all_predictions.append(predicted.cpu())
            all_barcodes.extend(barcodes) 


    # Compute Hamming Loss
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    hamming_loss_value = hamming_loss(all_labels.numpy(), all_predictions.numpy())

    # Compute final metrics
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    print(f"Hamming Loss: {hamming_loss_value:.4f}")
    results2csv(all_barcodes, all_predictions, dataset, columns)