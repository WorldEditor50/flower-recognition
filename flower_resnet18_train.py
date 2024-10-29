import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from imageutils.ImageDataset import ImageDataset
from model.resnet18 import ResNet18
from torchvision.models import resnet18


"""
model: Sequential(
  (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
  (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (5): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (6): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (7): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (8): AdaptiveAvgPool2d(output_size=(1, 1))
  (9): Linear(in_features=512, out_features=5, bias=True)
)
"""
image_size = 224
image_batch_size = 128
image_path = "D:/home/deeplearning/dataset/flowers"
model_path = './flower_resnet18_param'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-5
max_epoch = 10000
num_of_class = 5
    
def evaluate(model, loader):
    correct = 0
    total = len(loader.dataset)
    for imgs, labels in loader:
        with torch.no_grad():
            x = imgs.to(device)
            yt = labels.to(device)
            y = model(x)
            predicts = y.argmax(dim=1)
        correct += torch.eq(predicts, yt).sum().float().item()
    return correct/total

def main():
    print("Device:", device)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # load data
    train_dataset = ImageDataset(image_path, image_size, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=image_batch_size, shuffle=True, num_workers=2)
    validate_dataset = ImageDataset(image_path, image_size, 'validate')
    validate_dataloader = DataLoader(validate_dataset, batch_size=image_batch_size, shuffle=True, num_workers=1)
    test_dataset = ImageDataset(image_path, image_size, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=image_batch_size, shuffle=True, num_workers=1)
    # model
    pretrained_model = resnet18(pretrained=False)
    #for param in pretrained_model.parameters():
    #    param.requires_grad = False
    pretrained_model.fc = torch.nn.Linear(pretrained_model.fc.in_features, num_of_class)
    #model = torch.nn.Sequential(*list(pretrained_model.children()), # [b, 512, 1, 1]
    #    torch.nn.Linear(1000, num_of_class)
    #)
    model = pretrained_model.to(device)
    #print('model:', model)

    param = os.path.join(model_path,'flower_resnet18.pth')
    if os.path.exists(param):
        model.load_state_dict(torch.load(param))
        test_acc = evaluate(model, test_dataloader)
        print('current test accuracy:', test_acc)
    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer
    optiminzer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # train
    for i, epoch in enumerate(range(max_epoch)):
        for _, (imgs, labels) in enumerate(tqdm(train_dataloader, desc='epoch={}'.format(i))):
            x = imgs.to(device)
            yt = labels.to(device)
            y = model(x)
            loss = criterion(y, yt)
            optiminzer.zero_grad()
            loss.backward()
            optiminzer.step()
        # validate
        if (i + 1)%100 == 0 or i == 0:
            valid_acc = evaluate(model, validate_dataloader)
            test_acc = evaluate(model, test_dataloader)
            print("validate accuracy:", valid_acc, ', test accuracy:', test_acc)
            postfix = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            modelFile = os.path.join(model_path, f'flower_resnet18_{postfix}.pth')
            torch.save(model.state_dict(), modelFile)

    return


if __name__ == '__main__':
    main()