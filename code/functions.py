# improvement 1: write results of training and testing into a .txt file and just
# print 'results stored in: <name>.txt

import torch
from colorama import Fore as F

def train_step(model, loss_fn, acc_fn, optimizer, dataloader, epochs) -> None:
    model.train()
    train_loss, train_acc = 0, 0
    for epoch in range(epochs):
        for X, y in dataloader:
            y = y.unsqueeze(dim=1)
            logits = model(X)
            pred = (torch.sigmoid(logits) > 0.5) 
            loss = loss_fn(logits.type(torch.float32), y.type(torch.float32))
            acc = acc_fn(pred, y) * 100
            train_loss += loss.item()
            train_acc += acc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_loss /= (len(dataloader) * epochs)
    train_acc /= (len(dataloader) * epochs)

    if not os.path.exists('!RESULTS/'):
        os.mkdir('!RESULTS/')

    with open('train_results.txt', 'w') as file:
        file.writeline('your results are positive :D')


#        print(f'Epoch: {F.BLUE}{epoch}{F.RESET} | Loss: {F.RED}{train_loss:.2f}{F.RESET} | Accuracy: {F.GREEN}{train_acc:.2f}{F.RESET}')

def test_step(model, loss_fn, acc_fn, dataloader) -> None:
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            y = y.unsqueeze(dim=1)
            logits = model(X)
            pred = (torch.sigmoid(logits) > 0.5).float()
            loss = loss_fn(logits.type(torch.float32), y.type(torch.float32))
            acc = acc_fn(pred, y) * 100
            test_loss += loss.item()
            test_acc += acc

    test_loss /= (len(dataloader) * epochs)
    test_acc /= (len(dataloader) * epochs)

#        print(f'{F.CYAN}MÝR TESTING{F.RESET}\nLoss: {F.RED}{test_loss:.2f}{F.RESET} | Accuracy: {F.GREEN}{test_acc:.2f}{F.RESET}')
