import torch.nn as nn
import torch 

class InputTransform(nn.Module):
    """Transforms applied prior to backbone
    Attributes:
        clamp_min: values less than clamp_min are clamped to clamp_min
        clamp_max: values greater than clamp_max are clamped to clamp_max
    """
    def __init__(self, clamp_min = 0 , clamp_max = 255):
        super().__init__()
        self.clamp_min = 0
        self.clamp_max = 255
    
    def forward(self, x):
        x = x.float()
        if self.clamp_min>0 or self.clamp_max<255:
            x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x-=self.clamp_min
        x=x/(self.clamp_max-self.clamp_min)
        x = 2*x-1
        return x
        
class Backbone(nn.Module):
    """Convolution backbone for a classifier
    Attributes:
        cnn_layers: nn.Module of convolutional layers
    """
    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
        nn.Conv2d(3,32, kernel_size=5, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64,128, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128,256, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AdaptiveMaxPool2d(output_size = (1,1)))
        
    def forward(self, x):
        x = self.cnn_layers(x)
        return x
        
class Classifier(nn.Module):
    """Classifier based on adding a linear classifier on top of a convolutional
    backbone
    Attributes:
        input_transform: nn.Module called on the input before the backbone
        frozen_backbone: setting True puts the backbone in eval mode and
        prevents caluclation of gradients on backbone parameters
        backbone: nn.Module of convolutional layers
        head: linear classifier acting on the backbone output
    """
    def __init__(self, output_features, clamp_min = 0,
                 clamp_max = 255, clr_head = False): 
        super().__init__()
        self.input_transform = InputTransform(clamp_min, clamp_max)
        self.frozen_backbone = False
        self.backbone = Backbone()
        if clr_head:
            self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), 
                nn.Linear(256, output_features))
        else:
            self.head = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(256, output_features))
        
    def forward(self, x):
        x = self.input_transform(x)
        x = self.backbone(x)
        x = self.head(x.flatten(start_dim = 1))
        return x
    
    def freeze_backbone(self):
        self.frozen_backbone = True
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def unfreeze_backbone(self):
        self.frozen_backbone = False
        self.backbone.train(mode = self.training)
        for p in self.backbone.parameters():
            p.requires_grad = True
        
    def load_backbone(self, backbone_state_dict):
        self.backbone.load_state_dict(backbone_state_dict)
        self.frozen_backbone = False
        
    def train(self, mode = True):
        super().train(mode)
        if self.frozen_backbone:
            self.backbone.eval()
        return self
    
    def parameters(self):
        if self.frozen_backbone:
            return self.head.parameters()
        else:
            return super().parameters()