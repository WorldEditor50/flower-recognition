import os
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18
model_path = './flower_resnet18_param'
device = 'cpu'
num_of_class = 5
img_size = 224
def main():
    # load model
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_of_class)
    model = model.to(device)
    param = os.path.join(model_path,'flower_resnet18.pth')
    if os.path.exists(param):
        model.load_state_dict(torch.load(param))
    else:
        print("failed to load model")
        return
    model.eval()
    # classify
    # 0 --> daisy, 1 --> dandelion, 2 --> rose, 3 --> sunflower, 4 --> tulip
    getImage = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # normalize with image net parameter
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            lambda x: torch.unsqueeze(x, 0)
                                 ])
    img = getImage("./sunflower.jpeg").to(device)
    predict = model(img).argmax(dim=1).sum().item()
    if predict == 0:
        print("daisy")
    elif predict == 1:
        print("dandelion")
    elif predict == 2:
        print("rose")
    elif predict == 3:
        print("sunflower")
    elif predict == 4:
        print("tulip")
    else:
        print("unknow")
    return

if __name__ == '__main__':
    main()