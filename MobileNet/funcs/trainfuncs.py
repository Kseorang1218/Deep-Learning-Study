import torch
from .visualize import Visualize_ConfusionMatrix
from .utils import Get_F1Score

def train(model, train_loader, optimizer, device, criterion, batch_size):
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
        prediction =  prediction.view_as(label)

    train_loss /= (len(train_loader.dataset) / batch_size)
    train_accuracy = correct / len(train_loader.dataset)

    return train_loss, train_accuracy
    
#test()
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
            prediction =  prediction.view_as(label)

    test_loss /= (len(test_loader.dataset) / batch_size)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def test_with_confusionmatrix_f1scores(model, test_loader, device, criterion, batch_size, epoch, root):

    y_test = []
    predictions = []

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
            prediction =  prediction.view_as(label)

            y_test.extend(label)
            predictions.extend(prediction)

    # confusion matrix 그림 저장
    Visualize_ConfusionMatrix(y_test, predictions, epoch, root)

    # F1 score 기록
    Get_F1Score(y_test, predictions, epoch, root)

    test_loss /= (len(test_loader.dataset) / batch_size)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy