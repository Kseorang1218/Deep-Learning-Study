from download import *
from databuilder import *
from dataset import *
from processing import *

from torchvision import transforms
import torch.nn as nn

from ResNet import resnet
from MobileNet import MobileNetV2
from EfficientNet import efficientnet

import funcs

df = download_mimii("../../../data/mimii", "0_dB")

params = {
    "batch_size":32,
    "epoch": 50,
    "learning_rate": 0.0001,
    "path": "./saved/mobilenet/epoch50"
}

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


train_df, val_df, test_df = split_dataframe(df)

train_data, train_label = build_from_dataframe(train_df, 4096, 4096//2, False)
val_data, val_label = build_from_dataframe(val_df, 4096, 4096//2, False)
test_data, test_label = build_from_dataframe(test_df, 4096, 4096//2, False)

transform = transforms.Compose([
    melspectogram(),
    ExpandDim(),
    # ReSize(3, 32, 32),
    NpToTensor()
])

train_dataset = NumpyDataset(data=train_data, label=train_label, 
                             transform=transform, target_transform=NpToTensor())
val_dataset = NumpyDataset(data=val_data, label=val_label, 
                             transform=transform, target_transform=NpToTensor())
test_dataset = NumpyDataset(data=test_data, label=test_label, 
                             transform=transform, target_transform=NpToTensor())

train_loader = get_dataloader(dataset=train_dataset,
                              batch_size=params["batch_size"],
                              shuffle=True)
val_loader = get_dataloader(dataset=val_dataset,
                              batch_size=params["batch_size"],
                              shuffle=False)
test_loader = get_dataloader(dataset=test_dataset,
                              batch_size=params["batch_size"],
                              shuffle=False)

model = MobileNetV2().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

train_acc_list = []
train_loss_list = []

test_acc_list = []
test_loss_list = []

for epoch in range(1, params["epoch"] + 1):
    train_loss, train_accuracy = funcs.train(model, train_loader, optimizer, DEVICE, criterion, params["batch_size"])
    train_acc_list.append(train_accuracy)
    train_loss_list.append(train_loss)
    print("[EPOCH: {}] \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.4f}".format(
        epoch, train_loss, train_accuracy))

    test_loss, test_accuracy = funcs.test_with_confusionmatrix_f1scores(model, test_loader, DEVICE, criterion, 
                                                                        params["batch_size"], epoch, root=f"{params['path']}/confusionmatrix")
    test_acc_list.append(test_accuracy)
    test_loss_list.append(test_loss)

    print("[EPOCH: {}] \tTest Loss: {:.4f}, \tTest Accuracy: {:.4f}\n".format(
        epoch, test_loss, test_accuracy))
    # 에폭당 학습률 확인(필요시)
    # l_r = [x["lr"] for x in optimizer.param_groups]
    # print("lr:{}\n".format(l_r))


funcs.DrawGraph(test_acc_list, test_loss_list, 'test', mode='test', root=params["path"])
funcs.DrawGraph(train_acc_list, train_loss_list, 'train', mode='train', root=params["path"])

max_test_acc = max(test_acc_list)
print('max_test_acc: ', max_test_acc)

torch.save(model.state_dict(), f"{params['path']}/model.pth")