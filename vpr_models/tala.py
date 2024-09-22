import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import os


class GeMPool(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    we add flatten and norm so that we can use it as one aggregation layer.
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        x = x.flatten(1)
        return F.normalize(x, p=2, dim=1)


class CrossNet(nn.Module):
    def __init__(self, output_height, output_width):
        super(CrossNet, self).__init__()
        self.output_height = output_height
        self.output_width = output_width

        self.conv1_h = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 12), stride=(1,2), padding=(5, 0))
        self.bn1_h = nn.BatchNorm2d(4)
        self.conv1_v = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(6, 6), stride=(1,2), padding=(5, 0))
        self.bn1_v = nn.BatchNorm2d(8)

        self.conv2_h = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 12), padding=(5, 0))
        self.bn2_h = nn.BatchNorm2d(8)
        self.conv2_v = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(6, 6), padding=(5, 0))
        self.bn2_v = nn.BatchNorm2d(16)

        self.conv3_h = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 12), padding=(5, 0))
        self.bn3_h = nn.BatchNorm2d(16)
        self.conv3_v = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(6, 6), padding=(5, 0))
        self.bn3_v = nn.BatchNorm2d(32)

        self.conv4_h = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 12), padding=(5, 0))
        self.bn4_h = nn.BatchNorm2d(8)
        self.conv4_v = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(6, 6), padding=(5, 0))
        self.bn4_v = nn.BatchNorm2d(16)

        self.conv5_h = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 12), padding=(5, 0))
        self.bn5_h = nn.BatchNorm2d(4)
        self.conv5_v = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(6, 6), padding=(5, 0))
        self.bn5_v = nn.BatchNorm2d(8)

        self.conv6_h = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 12), padding=(5, 0))
        self.bn6_h = nn.BatchNorm2d(4)
        self.conv6_v = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(6, 6), padding=(5, 0))
        self.bn6_v = nn.BatchNorm2d(8)
        
        # Adaptive Average Pooling to adjust the output size
        self.aap = nn.AdaptiveAvgPool2d((output_height, output_width))
        
        # Final convolution to get the required output channels (3)
        # self.final_conv = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1) # for version1
        self.final_conv = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(3)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_h = self.conv1_h(x)
        x_h = self.bn1_h(x_h)
        x_h = self.relu(x_h)
        # print(f"shape of x_h is {x_h.shape}")

        x_h = self.conv2_h(x_h)
        x_h = self.bn2_h(x_h)
        x_h = self.relu(x_h)
        # print(f"shape of x_h is {x_h.shape}")

        x_h = self.conv3_h(x_h)
        x_h = self.bn3_h(x_h)
        x_h = self.relu(x_h)
        # print(f"shape of x_h is {x_h.shape}")

        x_h = self.conv4_h(x_h)
        x_h = self.bn4_h(x_h)
        x_h = self.relu(x_h)
        # print(f"shape of x_h is {x_h.shape}")

        x_h = self.conv5_h(x_h)
        x_h = self.bn5_h(x_h)
        x_h = self.relu(x_h)
        # print(f"shape of x_h is {x_h.shape}")

        x_h = self.conv6_h(x_h)
        x_h = self.bn6_h(x_h)
        x_h = self.relu(x_h)
        # print(f"shape of x_h is {x_h.shape}")

        x_h = self.aap(x_h)
        # print(f"shape of x_h is {x_h.shape}")

        x = self.final_conv(x_h)
        x = self.final_bn(x)
        x = self.relu(x)
        # print(f"shape of x is {x.shape}")

        return x

class ResNet(nn.Module):
    def __init__(self,
                 model_name='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 ):
        """Class representing the resnet backbone used in the pipeline
        we consider resnet network as a list of 5 blocks (from 0 to 4),
        layer 0 is the first conv+bn and the other layers (1 to 4) are the rest of the residual blocks
        we don't take into account the global pooling and the last fc

        Args:
            model_name (str, optional): The architecture of the resnet backbone to instanciate. Defaults to 'resnet50'.
            pretrained (bool, optional): Whether pretrained or not. Defaults to True.
            layers_to_freeze (int, optional): The number of residual blocks to freeze (starting from 0) . Defaults to 2.
            layers_to_crop (list, optional): Which residual layers to crop, for example [3,4] will crop the third and fourth res blocks. Defaults to [].

        Raises:
            NotImplementedError: if the model_name corresponds to an unknown architecture. 
        """
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = 'IMAGENET1K_V1'
        else:
            weights = None

        if 'swsl' in model_name or 'ssl' in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models', model_name)
        else:
            if 'resnext50' in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif 'resnet50' in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif '101' in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif '152' in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif '34' in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif '18' in model_name:
                # self.model = torchvision.models.resnet18(pretrained=False)
                self.model = torchvision.models.resnet18(weights=weights)
            elif 'wide_resnet50_2' in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError(
                    'Backbone architecture not recognized!')

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if '34' in model_name or '18' in model_name:
            out_channels = 512
            
        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x


class TALANet(nn.Module):
    def __init__(self,
                #---- cross
                output_height=320,
                output_width=320,

                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=2,
                layers_to_crop=[],  
                
                #---- Aggregator
                agg_arch='GeM', #CosPlace, NetVLAD, GeM
                agg_config={}
                ):
        super(TALANet, self).__init__()
        self.cross = CrossNet(output_height, output_width)
        self.backbone = ResNet()
        self.aggregation = GeMPool()

    def forward(self, x):
        x = T.Resize((256, 1024), interpolation=T.InterpolationMode.BILINEAR)(x)
        x = self.cross(x)
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_tala():
    file_path = "E:/TALA/VPR-methods-evaluation-master/VPR-methods-evaluation-master/trained_models/tala/best_perform.pth.tar"
    model_param = torch.load(file_path)

    model = TALANet()
    model.load_state_dict(model_param['state_dict'])
    model = model.eval()

    return model