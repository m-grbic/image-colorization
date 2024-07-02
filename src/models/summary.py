from torchvision import models
from torchsummary import summary

from model import ImageColorizerSE

model = ImageColorizerSE(backbone_name="resnext", freeze_backbone=True)
summary(model, (3, 224, 224))
