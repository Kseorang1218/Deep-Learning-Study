import torch
from torchvision import transforms, datasets
import torch.nn as nn

from model import Model

import funcs

'''
파라미터 정의 
'''
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.0001
SAVINGROOT = './saved/epoch3'



'''
GPU 사용 
'''
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')



'''
학습
'''
# 데이터셋 로드
train_dataset = datasets.CIFAR10(root = "../../data/cifar10/",
                                 train = True,
                                 download = True,
                                 transform=transforms.ToTensor())

test_dataset = datasets.CIFAR10(root = "../../data/cifar10/",
                                train = False,
                                transform=transforms.ToTensor())

test_dataset_for_visualize = datasets.CIFAR10(root = "../../data/cifar10/",
                                train = False,
                                transform=transforms.ToTensor())

# 전처리
train_mean, train_std, test_mean, test_std = funcs.get_mean_std(train_dataset, test_dataset)
train_transform, test_transform = funcs.do_transform(train_mean, train_std, test_mean, test_std)
train_dataset.transform = train_transform
test_dataset.transform = test_transform

# 데이터로더 정의
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = BATCH_SIZE,
                                            shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = BATCH_SIZE,
                                          shuffle = False)

test_loader_for_visualize = torch.utils.data.DataLoader(dataset = test_dataset_for_visualize,
                                          batch_size = BATCH_SIZE,
                                          shuffle = False)

# 모델, optimizer, loss function, scheduler 정의
model = Model().EfficientNetb0(num_classes=10).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

# train, test의 acc와 loss를 담을 리스트
train_acc_list = []
train_loss_list = []

test_acc_list = []
test_loss_list = []

# 학습 진행
for epoch in range(1, EPOCHS + 1):
    train_loss, train_accuracy = funcs.train(model, train_loader, optimizer, DEVICE, criterion, BATCH_SIZE)
    train_acc_list.append(train_accuracy)
    train_loss_list.append(train_loss)
    print("[EPOCH: {}] \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.4f}".format(
        epoch, train_loss, train_accuracy))

    test_loss, test_accuracy = funcs.test_with_confusionmatrix_f1scores(model, test_loader, DEVICE, criterion, BATCH_SIZE, epoch, f"{SAVINGROOT}/ConfusionMatrix")
    test_acc_list.append(test_accuracy)
    test_loss_list.append(test_loss)

    print("[EPOCH: {}] \tTest Loss: {:.4f}, \tTest Accuracy: {:.4f}\n".format(
        epoch, test_loss, test_accuracy))
    # 에폭당 학습률 확인(필요시)
    # l_r = [x["lr"] for x in optimizer.param_groups]
    # print("lr:{}\n".format(l_r))

# 그래프 그리고 저장
funcs.DrawGraph(test_acc_list, test_loss_list, 'test', mode='test', root=SAVINGROOT)
funcs.DrawGraph(train_acc_list, train_loss_list, 'train', mode='train', root=SAVINGROOT)

# 최대 test 정확도 출력
max_test_acc = max(test_acc_list)
print('max_test_acc: ', max_test_acc)

# 모델 저장
torch.save(model.state_dict(), f"{SAVINGROOT}/model.pth")

# 학습 결과 시각화
# model.load_state_dict(model.state_dict())
# funcs.visualize_model(model, SAVINGROOT, DEVICE, test_loader_for_visualize, train_dataset)