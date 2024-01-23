import torch
from torchvision import transforms, datasets
import torch.nn as nn

from model import MobileNetV2 # 내가 구현한 모델

# 이미지 확인
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np

'''
함수 정의 
'''
'''데이터 다루는 함수'''
# 데이터 transform 하는 함수
def do_transform(train_mean, train_std, test_mean, test_std):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([train_mean[0], train_mean[1], train_mean[2]], [train_std[0], train_std[1], train_std[2]]), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomChoice([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomAffine(
                degrees=15, translate=(0.2, 0.2),
                scale=(0.8, 1.2), shear=15)]),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([test_mean[0], test_mean[1], test_mean[2]], [test_std[0], test_std[1], test_std[2]]),
        ])

    return train_transform, test_transform

# normalize 위해 mean, std 구하는 함수
def do_mean_std(train_data, test_data):
    train_mean_rgb = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_data]
    train_std_rgb = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_data]

    train_mean_r = np.mean([m[0] for m in train_mean_rgb])
    train_mean_g = np.mean([m[1] for m in train_mean_rgb])
    train_mean_b = np.mean([m[2] for m in train_mean_rgb])

    train_std_r = np.mean([s[0] for s in train_std_rgb])
    train_std_g = np.mean([s[1] for s in train_std_rgb])
    train_std_b = np.mean([s[2] for s in train_std_rgb])

    test_mean_rgb = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in test_data]
    test_std_rgb = [np.std(x.numpy(), axis=(1, 2)) for x, _ in test_data]

    test_mean_r = np.mean([m[0] for m in test_mean_rgb])
    test_mean_g = np.mean([m[1] for m in test_mean_rgb])
    test_mean_b = np.mean([m[2] for m in test_mean_rgb])

    test_std_r = np.mean([s[0] for s in test_std_rgb])
    test_std_g = np.mean([s[1] for s in test_std_rgb])
    test_std_b = np.mean([s[2] for s in test_std_rgb])

    train_mean = [train_mean_r, train_mean_g, train_mean_b]
    train_std = [train_std_r, train_std_g, train_std_b]
    test_mean = [test_mean_r, test_mean_g, test_mean_b]
    test_std = [test_std_r, test_std_g, test_std_b]

    return train_mean, train_std, test_mean, test_std



'''그리는 함수'''
# 그래프 그리는 함수
def DrawGraph(acc_list, loss_list, title, mode, path):
    '''
    acc_list: train acc list or test acc list
    loss_list: train loss list or test loss list
    title: 그래프 제목
    mode: train인지 valid인지
    path: 그림 저장할 경로
    '''
    mode = mode.lower()
    valid_modes = ['train', 'test']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    x = np.arange(len(acc_list))

    fig, ax1 = plt.subplots()
    ax1.plot(x, acc_list, label= mode+' acc', color='blue')
    ax1.set_ylabel('acc')
    ax2 = ax1.twinx()
    ax2.plot(x, loss_list, label= mode+' loss', color='yellow')
    ax2.set_ylabel('loss')
    ax1.set_xlabel('epochs')

    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2
    ax1.legend(lines, labels, loc="upper left")
    plt.savefig(path)

# 학습 결과 이미지
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, num_images=9):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader_for_visualize):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if images_so_far < num_images:
                    row = images_so_far // 3
                    col = images_so_far % 3
                    axs[row, col].imshow(inputs.cpu().data[j].numpy().transpose((1, 2, 0)))
                    axs[row, col].axis('off')
                    axs[row, col].set_title('predicted: {}'.format(train_dataset.classes[preds[j]]))
                    plt.savefig('./result.png')
                images_so_far += 1

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


'''학습에 사용되는 함수'''
#train()
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

    test_loss /= (len(test_loader.dataset) / batch_size)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy




'''
파라미터 정의 
'''
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001
PATH = './saved/'



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
train_dataset = datasets.CIFAR10(root = "../data/cifar10/",
                                 train = True,
                                 download = True,
                                 transform=transforms.ToTensor())

test_dataset = datasets.CIFAR10(root = "../data/cifar10/",
                                train = False,
                                transform=transforms.ToTensor())

test_dataset_for_visualize = datasets.CIFAR10(root = "../data/cifar10/",
                                train = False,
                                transform=transforms.ToTensor())

# 전처리
train_mean, train_std, test_mean, test_std = do_mean_std(train_dataset, test_dataset)
train_transform, test_transform = do_transform(train_mean, train_std, test_mean, test_std)
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
model = MobileNetV2().to(DEVICE)
# from torchvision import models
# models.mobilenet_v2
# model = models.mobilenet_v2().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

# train, test의 acc와 loss를 담을 리스트
train_acc_list = []
train_loss_list = []

test_acc_list = []
test_loss_list = []

# 학습 진행
for epoch in range(1, EPOCHS + 1):
    train_loss, train_accuracy = train(model, train_loader, optimizer, DEVICE, criterion, epoch, BATCH_SIZE)
    train_acc_list.append(train_accuracy)
    train_loss_list.append(train_loss)
    print("[EPOCH: {}] \tTrain Loss: {:.4f}, \tTrain Accuracy: {:.4f}".format(
        epoch, train_loss, train_accuracy))

    test_loss, test_accuracy = test(model, test_loader, DEVICE, criterion, BATCH_SIZE)
    test_acc_list.append(test_accuracy)
    test_loss_list.append(test_loss)

    print("[EPOCH: {}] \tTest Loss: {:.4f}, \tTest Accuracy: {:.4f}\n".format(
        epoch, test_loss, test_accuracy))
    # 에폭당 학습률 확인(필요시)
    # l_r = [x["lr"] for x in optimizer.param_groups]
    # print("lr:{}\n".format(l_r))

# 그래프 그리고 저장
DrawGraph(test_acc_list, test_loss_list, 'test', mode='test', path=PATH+'test.png')
DrawGraph(train_acc_list, train_loss_list, 'train', mode='train', path=PATH+'train.png')
# 최대 test 정확도 출력
max_test_acc = max(test_acc_list)
print('max_test_acc: ', max_test_acc)

# 모델 저장
torch.save(model.state_dict(), PATH+'model.pth')

# 학습 결과 시각화
model.load_state_dict(model.state_dict())
visualize_model(model)