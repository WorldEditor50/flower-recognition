import os
import torch
from torchvision.models import resnet18
from model.resnet18 import ResNet18
model_path = './resnet18_param'
device = 'cpu'
num_of_class = 5
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
    # create a dummy input tensor
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    onnx_path = 'flower_resnet18.onnx'
    torch.onnx.export(model, x, onnx_path, verbose=True, input_names=['input'], output_names=['output'])
    print(f'Model exported to {onnx_path}')
    return

def main2():
    # load model
    model = ResNet18(num_of_class).to(device)
    param = os.path.join(model_path,'resnet18.pth')
    if os.path.exists(param):
        model.load_state_dict(torch.load(param))
    else:
        print("failed to load model")
        return
    model.eval()
    # create a dummy input tensor
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    onnx_path = 'resnet18.onnx'
    torch.onnx.export(model, x, onnx_path, verbose=True, input_names=['input'], output_names=['output'])
    print(f'Model exported to {onnx_path}')
    return

if __name__ == '__main__':
    main2()