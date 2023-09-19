import torch, torch.nn as nn, torchmetrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from colorama import Fore
from model import Mojmyr

# Set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# CHECKLIST:
# 1. Data augmentation - use different techniques of augmentation to make the dataset larger
# 2. Visualize the results like in the course

# Save the paths of the dataset's train and test folders
data_path = Path("data/")
train_dir = data_path / "train"
test_dir = data_path / "test"

# Define data transform for dataset
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# Turn the images into PyTorch-compatible datasets using ImageFolder
train_data = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform, target_transform=None)

# Making dataloaders out of train and test datasets with the batch size of 1 (#mýr dataset isn't that big)
BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

def train_step(model, loss_fn, acc_fn, optimizer, dataloader, epochs):
    """
    Trains a model for a binary classification task, calculating both loss and accuracy
    Args:
        model: the model that will be trained
        loss_fn: loss function, should be BCEWithLogitsLoss
        acc_fn: accuracy function (ideally from torchmetrics)
        optimizer: optimizer from torch.optim
        dataloader: dataloader for the data the model will be trained on
        epochs: the number of times the model will run through the entire dataloader
  
    Returns:
        Each epoch the print out of the current epoch number, loss value, accuracy value (all coloured)          
    """
    model.train()
    train_loss, train_acc = 0, 0
    for epoch in range(epochs):
        for X, y in dataloader:
            print(X.shape)
            quit()
            X, y = X.to(device), y.to(device)
            y = y.unsqueeze(dim=1)
            logits = model(X)
            pred = (torch.sigmoid(logits) > 0.5) # Convert logits to probabilites using sigmoid function, then to labels by setting a 0.5 treshold
            loss = loss_fn(logits.type(torch.float32), y.type(torch.float32))
            acc = acc_fn(pred, y) * 100
            train_loss += loss.item()
            train_acc += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        print(f'Epoch: {Fore.BLUE}{epoch}{Fore.RESET} | Loss: {Fore.RED}{train_loss:.2f}{Fore.RESET} | Accuracy: {Fore.GREEN}{train_acc:.2f}{Fore.RESET}')
    return model

def test_step(model, loss_fn, acc_fn, dataloader):
    """
    Tests a model on a binary classification task, calculating both loss and accuracy
    Args:
        model: the model that will be tested
        loss_fn: loss function, should be BCEWithLogitsLoss
        acc_fn: accuracy function (ideally from torchmetrics)
        dataloader: dataloader for the data the model will be tested on
  
    Returns:
        At the end of testing; the measured loss and accuracy value (all coloured)          
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y = y.unsqueeze(dim=1)
            logits = model(X)
            pred = (torch.sigmoid(logits) > 0.5).float() # Convert logits to probabilites using sigmoid function, then to labels by setting a 0.5 treshold
            loss = loss_fn(logits.type(torch.float32), y.type(torch.float32))
            acc = acc_fn(pred, y) * 100
            test_loss += loss.item()
            test_acc += acc

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        print(f'{Fore.CYAN}MÝR TESTING{Fore.RESET}\nLoss: {Fore.RED}{test_loss:.2f}{Fore.RESET} | Accuracy: {Fore.GREEN}{test_acc:.2f}{Fore.RESET}')

model_0 = Mojmyr(input_shape=3, hidden_units=10, output_shape=1).to(device) # Input 3 because RGB channels; output 1 because this is binary classification
loss_fn = nn.BCEWithLogitsLoss().to(device)
acc_fn = torchmetrics.Accuracy(task='binary').to(device)
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

EPOCHS = 15
model_0 = train_step(model=model_0, loss_fn=loss_fn, acc_fn=acc_fn, optimizer=optimizer, dataloader=train_dataloader, epochs=EPOCHS)
test_step(model=model_0, loss_fn=loss_fn, acc_fn=acc_fn, dataloader=test_dataloader)

torch.save(model_0.state_dict(), 'model_0_state_dict.pth')