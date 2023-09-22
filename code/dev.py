import torch, torch.nn as nn, torchmetrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from model import Mojmyr
import functions as f

# Set up device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Save the paths of the dataset's train and test folders - not included on GitHub using .gitignore because GDPR reasons
data_path = Path("code/data/")
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

model_0 = Mojmyr(input_shape=3, hidden_units=30, output_shape=1).to(device) # Input 3 because RGB channels; output 1 because this is binary classification
loss_fn = nn.BCEWithLogitsLoss().to(device)
acc_fn = torchmetrics.Accuracy(task='binary').to(device)
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

EPOCHS = 15
f.train_step(model=model_0, loss_fn=loss_fn, acc_fn=acc_fn, optimizer=optimizer, dataloader=train_dataloader, epochs=EPOCHS, device=device)
f.test_step(model=model_0, loss_fn=loss_fn, acc_fn=acc_fn, dataloader=test_dataloader, device=device)

torch.save(model_0.state_dict(), 'code/!model_0_state_dict.pth')