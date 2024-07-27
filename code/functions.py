import torch

def train_step(model, loss_fn, acc_fn, optimizer, dataloader, epochs, date) -> None:
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

        store_results('train_results', epoch, train_loss, train_acc, date)

def test_step(model, loss_fn, acc_fn, dataloader, date) -> None:
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
        
        store_results('test_results', epoch, test_loss, test_acc, date)

def store_results(filename, epoch, loss, acc, date) -> None:
    if not os.path.exists('!Results/'):
        os.mkdir('!Results/')

    txt_file = f'!Results/{date}{filename}.txt'

    with open(txt_file, 'w') as file:
        file.write(f'Epoch: {epoch} | Loss: {loss} | Accuracy: {acc}')
        file.write('\n')
