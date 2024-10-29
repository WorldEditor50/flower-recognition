import os
import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from imageutils.ImageDataset import ImageDataset
from model.resnet18 import ResNet18

image_size = 224
image_batch_size = 128
image_path = "D:/home/deeplearning/dataset/flowers"
model_path = './resnet18_param'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-3
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
    model = ResNet18(num_of_class).to(device)
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