import torch

def train(model, train_loader, optimizer, device, criterion, epoch, batch_size):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.max(1, keepdim = True)[1]
        correct += prediction.eq(label.view_as(prediction)).sum().item()

    train_loss /= (len(train_loader.dataset) / batch_size)
    train_accuracy = correct / len(train_loader.dataset)
    return train_loss, train_accuracy
    # train_acc_list.append(train_accuracy)
    # train_loss_list.append(train_loss)
    # print("[EPOCH: {}] \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.4f}".format(
    #     epoch, train_loss, train_accuracy))
    

def test(model, test_loader, device, criterion, batch_size):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= (len(test_loader.dataset) / batch_size)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy





# for epoch in range(1, EPOCHS + 1):
#     train(model, train_loader, optimizer, log_interval = 200, device=DEVICE, criterion=criterion, epoch=epoch)
#     test_loss, test_accuracy = test(model, test_loader, device=DEVICE, criterion=criterion, batch_size=BATCH_SIZE, scheduler=scheduler)
#     print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %".format(
#         epoch, test_loss, test_accuracy))