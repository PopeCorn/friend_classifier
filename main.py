import torch, torch.nn as nn, torchmetrics
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from colorama import Fore

# CHECKLIST:
# 1. Data augmentation - use different techniques of augmentation to make the dataset larger
# 2. Background removal - try using the dataset with and without background to see how it affects performance
# 3. Learning rate scheduler - try this method to change the learning rate throughout the training process
# 4. Visualize the results like in the course

# Save the paths of the image, train, test folders
image_path = Path("images_without_bg/")
train_dir = image_path / "train"
test_dir = image_path / "test"

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

class Mojmyr(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        # Copy TinyVGG structure, modify it slightly for this specific case
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, 3, 1, 1), 
            nn.ReLU(), 
            nn.Conv2d(hidden_units, hidden_units, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*14*14, out_features=output_shape)
        )

    # Required forward method that takes the input 'x' through all the conv_blocks and the classifier, returning logits because of the last Linear layer
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

def train_step(model: nn.Module, loss_fn: nn.Module, acc_fn: torchmetrics.Accuracy, optimizer: torch.optim, dataloader: DataLoader, epochs):
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

def test_step(model: nn.Module, loss_fn: nn.Module, acc_fn: torchmetrics.Accuracy, dataloader: DataLoader):
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

model_0 = Mojmyr(input_shape=3, hidden_units=10, output_shape=1) # Input 3 because RGB channels; output 1 because this is binary classification
loss_fn = nn.BCEWithLogitsLoss()
acc_fn = torchmetrics.Accuracy(task='binary')
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01) # Will make learning rate scheduler

EPOCHS = 22
train_step(model=model_0, loss_fn=loss_fn, acc_fn=acc_fn, optimizer=optimizer, dataloader=train_dataloader, epochs=EPOCHS)
test_step(model=model_0, loss_fn=loss_fn, acc_fn=acc_fn, dataloader=test_dataloader)