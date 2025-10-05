import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # Load the VGG16 model with pretrained weights
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        
        # We only need the feature extraction part of VGG
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        
        # Freeze the layers, as we don't want to train VGG
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        
        # Register normalization constants as buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # The input images are expected to be in (B, C, H, W) format, range [0, 1]
        
        # Normalize the images using ImageNet stats
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        loss = 0.0
        x = input
        y = target
        
        # Pass the images through the VGG blocks and compare features
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            # We calculate L1 loss on the feature maps
            loss += torch.nn.functional.l1_loss(x, y)
            
        return loss