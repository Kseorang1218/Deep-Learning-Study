# trainer.py

import torch
import numpy as np

class Trainer:
    def __init__(self, model, loss, optimizer, device):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device

    def training_step(self, train_loader):
        train_loss_list = []

        for batch_idx, (data, laebl) in enumerate(train_loader):
            x = data.to(self.device)
            output = self.model(x)
            loss = self.loss(x, output)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss_list.append(loss.item())

        return train_loss_list

    def train(self, epoch, train_loader):
        self.model.train()
        for epoch in range(0, epoch + 1):
            train_loss_list = self.training_step(train_loader)
            print(f"[EPOCH: {epoch}] \tTrain Loss: {np.mean(train_loss_list):.4f}")
            




    # def eval(self):

    # def test(self):





def train(model, train_loader, optimizer, criterion, batch_size):
    model.train()
    train_loss = 0
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.float()
        label = label
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(image, output)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= (len(train_loader.dataset) / batch_size)

    return train_loss
    
#test()
def test(model, test_loader, criterion, batch_size):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image
            label = label
            output = model(image)
            test_loss += criterion(output, label).item()

    test_loss /= (len(test_loader.dataset) / batch_size)
    return test_loss
