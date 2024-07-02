from torchvision import models
from torchsummary import summary

from image_colorizer_classificator import ImageColorizerClassificator

model = ImageColorizerClassificator(backbone_name="resnext", freeze_backbone=True)
summary(model, (3, 224, 224))
