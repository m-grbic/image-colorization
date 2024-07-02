import torch
import torch.nn as nn
import torchvision.models as models
import timm


class ImageColorizerSE(nn.Module):

    def __init__(self, backbone_name: str, pretrained: bool = True, freeze_backbone: bool = False, upsampling_method: str = 'deconv'):
        super().__init__()

        # Dictionary to map backbone names to their respective models
        self.backbone_name = backbone_name.lower()
        backbone_dict = {
            'resnet': nn.Sequential(*list(models.resnet50(weights='DEFAULT' if pretrained else None).children())[:-2]),  # Remove the last two layers,
            'resnext': nn.Sequential(*list(models.resnext50_32x4d(weights='DEFAULT' if pretrained else None).children())[:-2]),  # Remove the last two layers
            'vit': nn.Sequential(*list(timm.create_model('vit_base_patch16_224', pretrained=pretrained).children())[:-2])  # Using torchvision for ViT
        }
        self.backbone = backbone_dict[self.backbone_name]

        # Freezing backbone if requested
        if freeze_backbone:
            print("Freezing backbone parameters.")
            for param in self.backbone.parameters():
                param.requires_grad = False
        print(f"Number od trainable parameters:", sum(p.numel() for p in self.backbone.parameters() if p.requires_grad))

        if self.backbone_name == 'vit':
            self.vit_to_conv = nn.Conv2d(768, 1024, kernel_size=1)  # ViT has an output of 768 channels, convert to 512

        # Upsampling layers
        if upsampling_method == 'up_conv':
            print("Upsampling method is upsample + convolution")
            #  Upsampling + Convolution | 11,230,491 trainable parameters
            self.upsample_7_to_14 = nn.Sequential(
                nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False),
                nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            self.upsample_14_to_64 = nn.Sequential(
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False),
                nn.Conv2d(256, 265, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(265),
                nn.ReLU(inplace=True)
            )
        elif upsampling_method == 'conv_up':
            print("Upsampling method is convolution + upsampling")
            # Convolution + Upsampling | 3,847,963 trainable parameters
            self.upsample_7_to_14 = nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False)
            )
            self.upsample_14_to_64 = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False),
                nn.Conv2d(512, 265, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(265),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
            )
        elif upsampling_method == 'deconv':
            print("Upsampling method is transposed convolution")
            # Transposed convolution | 19,962,907 trainable parameters
            self.upsample_7_to_14 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=2048, 
                    out_channels=512, 
                    kernel_size=4, 
                    stride=2, 
                    padding=1, 
                    output_padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            self.upsample_14_to_64 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=512, 
                    out_channels=256, 
                    kernel_size=4, 
                    stride=2, 
                    padding=1, 
                    dilation=2,
                    output_padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256, 
                    out_channels=265, 
                    kernel_size=4, 
                    stride=2, 
                    padding=1, 
                    output_padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(265),
                nn.ReLU(inplace=True)
            )
        else:
            raise RuntimeError(f"Upsampling method not recognized: {upsampling_method}")

    def forward(self, x):
        if self.backbone_name == 'vit':
            x = self.backbone(x)
            B, N, C = x.shape  # (batch_size, 196, 768)
            H = W = int(N ** 0.5)  # H = W = 14 for 196 patches
            x = x.permute(0, 2, 1).view(B, C, H, W)  # (batch_size, 768, 14, 14)
            x = self.vit_to_conv(x)  # (batch_size, 265, 14, 14)
        else:
            x = self.backbone(x)

        if self.backbone_name in ('resnet', 'resnext'):
            x = self.upsample_7_to_14(x)
        
        x = self.upsample_14_to_64(x)
        x = torch.nn.functional.softmax(x, dim=1)

        return x
    