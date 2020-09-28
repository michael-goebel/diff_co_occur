from utils.detection.torch_pre_proc import CoOccurWithNorm, DFTWithNorm, ImageNetNorm

import torch, torch.nn as nn, torchvision


def get_model(name,method,pretrained=True):

    if name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(4096,2)

    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(512,2)

    if name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(2048,2)

    if name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(2048,2)

    if name == 'resnet152':
        model = torchvision.models.resnet152(pretrained=pretrained)
        model.fc = nn.Linear(2048,2)

    if name == 'resnext50':
        model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        model.fc = nn.Linear(2048,2)

    if name == 'inception_v3':
        model = torchvision.models.inception_v3(pretrained=pretrained)
        model.aux_logits = False
        model.fc = nn.Linear(2048,2)

    if name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(1280,2)

    if name == 'squeezenet':
        model = torchvision.models.squeezenet1_1(pretrained=pretrained)
        model.classifier[1] = nn.Conv2d(512,2,kernel_size=(1,1), stride=(1,1))


    if method == 'co_occur': pre_proc = CoOccurWithNorm()
    if method == 'dft': pre_proc = DFTWithNorm()
    if method == 'direct': pre_proc = ImageNetNorm()
    if method == 'cband_cc':
        pre_proc = CoOccurWithNorm(cband=True)
        model.conv1 = torch.nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return nn.Sequential(pre_proc,model)


