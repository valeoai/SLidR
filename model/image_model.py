import os
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
from model.modules.resnet_encoder import resnet_encoders
import model.modules.dino.vision_transformer as dino_vit

_MEAN_PIXEL_IMAGENET = [0.485, 0.456, 0.406]
_STD_PIXEL_IMAGENET = [0.229, 0.224, 0.225]


def adapt_weights(architecture):
    if architecture == "imagenet" or architecture is None:
        return

    weights_url = {
        "moco_v2": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar",
        "moco_v1": "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar",
        "swav": "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
        "deepcluster_v2": "https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar",
        "dino": "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
    }

    if not os.path.exists(f"weights/{architecture}.pt"):
        r = requests.get(weights_url[architecture], allow_redirects=True)
        os.makedirs("weights", exist_ok=True)
        with open(f"weights/{architecture}.pt", 'wb') as f:
            f.write(r.content)

    weights = torch.load(f"weights/{architecture}.pt")

    if architecture == "obow":
        return weights["network"]

    if architecture == "pixpro":
        weights = {
            k.replace("module.encoder.", ""): v
            for k, v in weights["model"].items()
            if k.startswith("module.encoder.")
        }
        return weights

    if architecture in ("moco_v1", "moco_v2", "moco_coco"):
        weights = {
            k.replace("module.encoder_q.", ""): v
            for k, v in weights["state_dict"].items()
            if k.startswith("module.encoder_q.") and not k.startswith("module.encoder_q.fc")
        }
        return weights

    if architecture in ("swav", "deepcluster_v2"):
        weights = {
            k.replace("module.", ""): v
            for k, v in weights.items()
            if k.startswith("module.") and not k.startswith("module.pro")
        }
        return weights

    if architecture == "dino":
        return weights


class Preprocessing:
    """
    Use the ImageNet preprocessing.
    """

    def __init__(self):
        normalize = T.Normalize(mean=_MEAN_PIXEL_IMAGENET, std=_STD_PIXEL_IMAGENET)
        self.preprocessing_img = normalize

    def __call__(self, image):
        return self.preprocessing_img(image)


class DilationFeatureExtractor(nn.Module):
    """
    Dilated ResNet Feature Extractor
    """

    def __init__(self, config, preprocessing=None):
        super(DilationFeatureExtractor, self).__init__()
        assert (
            config["images_encoder"] == "resnet50"
        ), "DilationFeatureExtractor is only available for resnet50"
        Encoder = resnet_encoders["resnet50"]["encoder"]
        params = resnet_encoders["resnet50"]["params"]
        params.update(replace_stride_with_dilation=[True, True, True])
        self.encoder = Encoder(**params)

        if config["image_weights"] == "imagenet":
            self.encoder.load_state_dict(model_zoo.load_url(model_urls["resnet50"]))

        weights = adapt_weights(architecture=config["image_weights"])
        if weights is not None:
            self.encoder.load_state_dict(weights)

        for param in self.encoder.parameters():
            param.requires_grad = False

        in1 = 2048

        self.decoder = nn.Sequential(
            nn.Conv2d(in1, config["model_n_out"], 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
        )
        self.preprocessing = preprocessing
        self.normalize_feature = config["normalize_features"]

    def forward(self, x):
        if self.preprocessing:
            x = self.preprocessing(x)
        x = self.decoder(self.encoder(x))
        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=1)
        return x


class PPKTFeatureExtractor(nn.Module):
    """
    PPKT baseline
    """

    def __init__(self, config, preprocessing=None):
        super(PPKTFeatureExtractor, self).__init__()
        Encoder = resnet_encoders[config["images_encoder"]]["encoder"]
        params = resnet_encoders[config["images_encoder"]]["params"]
        self.encoder = Encoder(**params)

        if config["image_weights"] == "imagenet":
            self.encoder.load_state_dict(model_zoo.load_url(model_urls[config["images_encoder"]]))

        if config["image_weights"] not in (None, "imagenet"):
            assert (
                config["images_encoder"] == "resnet50"
            ), "{} weights are only available for resnet50".format(
                config["images_weights"]
            )
            weights = adapt_weights(architecture=config["image_weights"])
            if weights is not None:
                self.encoder.load_state_dict(weights)

        for param in self.encoder.parameters():
            param.requires_grad = False

        if config["images_encoder"] == "resnet18":
            in1 = 512
        elif config["images_encoder"] == "resnet50":
            in1 = 2048

        self.decoder = nn.Sequential(
            nn.Conv2d(in1, config["model_n_out"], 1),
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=True),
        )
        self.preprocessing = preprocessing
        self.normalize_feature = config["normalize_features"]

    def forward(self, x):
        if self.preprocessing:
            x = self.preprocessing(x)
        x = self.decoder(self.encoder(x))
        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=1)
        return x


class DinoVitFeatureExtractor(nn.Module):
    """
    DINO Vision Transformer Feature Extractor.
    """
    def __init__(self, config, preprocessing=None):
        super(DinoVitFeatureExtractor, self).__init__()
        dino_models = {
            "vit_small_p16": ("vit_small", 16, 384),
            "vit_small_p8": ("vit_small", 8, 384),
            "vit_base_p16": ("vit_base", 16, 768),
            "vit_base_p8": ("vit_base", 8, 768),
        }
        assert (
            config["images_encoder"] in dino_models.keys()
        ), f"DilationFeatureExtractor is only available for {dino_models.keys()}"

        model_name, patch_size, embed_dim = dino_models[config["images_encoder"]]

        print("Use Vision Transformer pretrained with DINO as the image encoder")
        print(f"==> model_name: {model_name}")
        print(f"==> patch_size: {patch_size}")
        print(f"==> embed_dim: {embed_dim}")

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.encoder = dino_vit.__dict__[model_name](patch_size=patch_size, num_classes=0)
        dino_vit.load_pretrained_weights(self.encoder, "", None, model_name, patch_size)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, config["model_n_out"], 1),
            nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=True),
        )
        self.preprocessing = preprocessing
        self.normalize_feature = config["normalize_features"]

    def forward(self, x):
        if self.preprocessing:
            x = self.preprocessing(x)
        batch_size, _, height, width = x.size()
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        f_height = height // self.patch_size
        f_width = width // self.patch_size

        x = self.encoder(x, all=True)
        # the output of x should be [batch_size x (1 + f_height * f_width) x self.embed_dim]
        assert x.size(1) == (1 + f_height * f_width)
        # Remove the CLS token and reshape the the patch token features.
        x = x[:, 1:, :].contiguous().transpose(1, 2).contiguous().view(batch_size, self.embed_dim, f_height, f_width)

        x = self.decoder(x)
        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=1)
        return x
