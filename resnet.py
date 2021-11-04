import torch 
import torch.nn as nn
import torchvision

class Identity(nn.Module):
    """Identity module used to remove layers from a network
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x
    
class InputTransform(nn.Module):
    """Transforms applied prior to backbone
    """
    def __init__(self, clamp_min = 0, clamp_max = 255):
        super().__init__()
        self.clamp_min = clamp_min  
        self.clamp_max = clamp_max
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(mean=self.means, 
                                                          std=self.stds)
    
    def forward(self, x):
        x = x.float()
        if self.clamp_min>0 or self.clamp_max<255:
            x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x-=self.clamp_min
        x=x/(self.clamp_max-self.clamp_min)
        x = self.normalize(x)
        return x
        
class Classifier(nn.Module):
    """Classifier based on adding a linear classifier on top of a CNN
    backbone
    Attributes:
        input_transform: nn.Module called on the input before the backbone
        frozen_backbone: setting True puts the backbone in eval mode and
        prevents caluclation of gradients on backbone parameters
        backbone: nn.Module of convolutional layers
        head: linear classifier acting on the backbone output
        use_head: if False the output layer is replaced with identity
    """
    def __init__(self, output_features = 1, clamp_min = 0,
                 clamp_max = 255, clr_head = False): 
        super().__init__()
        self.input_transform = InputTransform(clamp_min, clamp_max)
        self.frozen_backbone = False
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, 512)
        if clr_head:
            self.head = nn.Sequential(nn.ReLU(), nn.Linear(512, 512),
                nn.ReLU(), nn.Linear(512, output_features))
        else:
            self.head = nn.Sequential(nn.ReLU(),
            nn.Dropout(p=0.5), nn.Linear(512, output_features))

    def forward(self, x):
        x = self.input_transform(x)
        x = self.backbone(x)
        x = self.head(x)
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