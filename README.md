# flower recognition

- 数据集

  https://www.kaggle.com/datasets/imsparsh/flowers-dataset

- 数据处理

- 训练

- 测试验证

- pytorch模型转onnx

- pytorch模型转pnnx

  - 导出TorchScript模型

    ```python
    import torch
    import torchvision.models as models
    from torchvision.models import resnet18
    import os
    model_path = './resnet18_param1'
    device = 'cpu'
    num_of_class = 5
    img_size = 224
    net = resnet18(pretrained=False)
    net.fc = torch.nn.Linear(net.fc.in_features, num_of_class)
    param = os.path.join(model_path,'flower_resnet18.pth')
    if os.path.exists(param):
        net.load_state_dict(torch.load(param))
    else:
        print("failed to load model")
    
    net.eval()
    
    x = torch.rand(1, 3, 224, 224)
    mod = torch.jit.trace(net, x)
    mod.save("flower_resnet18.pt")
    ```

    

  - 转换

    ```shell
    .\pnnx.exe .\flower_resnet18.pt inputshape=[1,3,224,224]
    ```

    



