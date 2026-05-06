import torch.nn as nn
from torchvision import models


class NIH_DenseNet121(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.model = models.densenet121(weights=None)

        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)