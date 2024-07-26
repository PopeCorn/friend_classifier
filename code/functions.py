import torch
from datetime import datetime

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

        train_loss /= len(dataloader)
        train_acc /= len(dataloader)

        store_results('train_results', epoch, train_loss, train_acc)

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

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        
        store_results('test_results', epoch, test_loss, test_acc)

def determine_time() -> str:
    now = datetime.now()
    date = now.strftime("%d/%m/%Y %H:%M:%S") # normal people format
    output = date.replace(" ", "|")
    return output

def store_results(filename, epoch, loss, acc) -> None:
    if not os.path.exists('!RESULTS/'):
        os.mkdir('!RESULTS/')

    date = determine_time()
    txt_file = f'!RESULTS/{date}{filename}.txt'

    with open(txt_file, 'w') as file:
        file.write(f'Epoch: {epoch} | Loss: {loss} | Accuracy: {acc}')
        file.write('\n')
