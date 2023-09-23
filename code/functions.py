import torch
from colorama import Fore as F

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
        print(f'Epoch: {F.BLUE}{epoch}{F.RESET} | Loss: {F.RED}{train_loss:.2f}{F.RESET} | Accuracy: {F.GREEN}{train_acc:.2f}{F.RESET}')

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
            y = y.unsqueeze(dim=1)
            logits = model(X)
            pred = (torch.sigmoid(logits) > 0.5).float() # Convert logits to probabilites using sigmoid function, then to labels by setting a 0.5 treshold
            loss = loss_fn(logits.type(torch.float32), y.type(torch.float32))
            acc = acc_fn(pred, y) * 100
            test_loss += loss.item()
            test_acc += acc

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        print(f'{F.CYAN}MÝR TESTING{F.RESET}\nLoss: {F.RED}{test_loss:.2f}{F.RESET} | Accuracy: {F.GREEN}{test_acc:.2f}{F.RESET}')
