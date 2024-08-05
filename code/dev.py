import torch, torch.nn as nn, torchmetrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from model import Friend
import functions as f

data_path = Path("data/") # dirs were deleted after the model was trained
train_dir = data_path / "train"
test_dir = data_path / "test"
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()])

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform, 
                                  target_transform=None)
test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=data_transform, 
                                 target_transform=None)

BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=BATCH_SIZE, 
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=BATCH_SIZE, 
                             shuffle=False)

now = datetime.now() # used in functions.py for naming files with results
date = now.strftime("%d/%m/%Y %H:%M:%S").replace(" ", "|")

friend_model = Friend(input_shape=3, hidden_units=100, output_shape=1)
loss_fn = nn.BCEWithLogitsLoss()
acc_fn = torchmetrics.Accuracy(task='binary')
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

EPOCHS = 12
f.train_step(model=friend_model, 
             loss_fn=loss_fn, 
             acc_fn=acc_fn, 
             optimizer=optimizer, 
             dataloader=train_dataloader, 
             epochs=EPOCHS,
             date=date)
f.test_step(model=friend_model, 
            loss_fn=loss_fn, 
            acc_fn=acc_fn, 
            dataloader=test_dataloader,
            date=date)

torch.save(friend_model.state_dict(), '!friend_model_state_dict.pth')
