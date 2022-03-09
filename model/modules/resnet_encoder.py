import torch.nn as nn

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck


class ResNetEncoder(ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(6):
            x = stages[i](x)
            features.append(x)

        return features[5]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


resnet_encoders = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "params": {
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "params": {
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
}
