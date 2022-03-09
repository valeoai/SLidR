import torch
import random
import numpy as np
from torchvision.transforms import InterpolationMode
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.functional import resize, resized_crop, hflip


class ComposeClouds:
    """
    Compose multiple transformations on a point cloud.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pc):
        for transform in self.transforms:
            pc = transform(pc)
        return pc


class Rotation_z:
    """
    Random rotation of a point cloud around the z axis.
    """

    def __init__(self):
        pass

    def __call__(self, pc):
        angle = np.random.random() * 2 * np.pi
        c = np.cos(angle)
        s = np.sin(angle)
        R = torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        pc = pc @ R.T
        return pc


class FlipAxis:
    """
    Flip a point cloud in the x and/or y axis, with probability p for each.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pc):
        for curr_ax in range(2):
            if random.random() < self.p:
                pc[:, curr_ax] = -pc[:, curr_ax]
        return pc


def make_transforms_clouds(config):
    """
    Read the config file and return the desired transformation on point clouds.
    """
    transforms = []
    if config["transforms_clouds"] is not None:
        for t in config["transforms_clouds"]:
            if t.lower() == "rotation":
                transforms.append(Rotation_z())
            elif t.lower() == "flipaxis":
                transforms.append(FlipAxis())
            else:
                raise Exception(f"Unknown transformation: {t}")
    if not len(transforms):
        return None
    return ComposeClouds(transforms)


class ComposeAsymmetrical:
    """
    Compose multiple transformations on a point cloud, and image and the
    intricate pairings between both (only available for the heavy dataset).
    Note: Those transformations have the ability to increase the number of
    images, and drastically modify the pairings
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pc, features, img, pairing_points, pairing_images, superpixels=None):
        for transform in self.transforms:
            pc, features, img, pairing_points, pairing_images, superpixels = transform(
                pc, features, img, pairing_points, pairing_images, superpixels
            )
        if superpixels is None:
            return pc, features, img, pairing_points, pairing_images
        return pc, features, img, pairing_points, pairing_images, superpixels


class ResizedCrop:
    """
    Resize and crop an image, and adapt the pairings accordingly.
    """

    def __init__(
        self,
        image_crop_size=(224, 416),
        image_crop_range=[0.3, 1.0],
        image_crop_ratio=(14.0 / 9.0, 17.0 / 9.0),
        image_interpolation=InterpolationMode.BILINEAR,
        crop_center=False,
    ):
        self.crop_size = image_crop_size
        self.crop_range = image_crop_range
        self.crop_ratio = image_crop_ratio
        self.img_interpolation = image_interpolation
        self.crop_center = crop_center

    def __call__(self, pc, features, images, pairing_points, pairing_images, superpixels):
        imgs = torch.empty(
            (images.shape[0], 3) + tuple(self.crop_size), dtype=torch.float32
        )
        if superpixels is not None:
            superpixels = superpixels.unsqueeze(1)
            sps = torch.empty(
                (images.shape[0],) + tuple(self.crop_size), dtype=torch.uint8
            )
        pairing_points_out = np.empty(0, dtype=np.int64)
        pairing_images_out = np.empty((0, 3), dtype=np.int64)
        if self.crop_center:
            pairing_points_out = pairing_points
            _, _, h, w = images.shape
            for id, img in enumerate(images):
                mask = pairing_images[:, 0] == id
                p2 = pairing_images[mask]
                p2 = np.round(
                    np.multiply(p2, [1.0, self.crop_size[0] / h, self.crop_size[1] / w])
                ).astype(np.int64)

                imgs[id] = resize(img, self.crop_size, self.img_interpolation)
                if superpixels is not None:
                    sps[id] = resize(
                        superpixels[id], self.crop_size, InterpolationMode.NEAREST
                    )

                p2[:, 1] = np.clip(0, self.crop_size[0] - 1, p2[:, 1])
                p2[:, 2] = np.clip(0, self.crop_size[1] - 1, p2[:, 2])
                pairing_images_out = np.concatenate((pairing_images_out, p2))

        else:
            for id, img in enumerate(images):
                successfull = False
                mask = pairing_images[:, 0] == id
                P1 = pairing_points[mask]
                P2 = pairing_images[mask]
                while not successfull:
                    i, j, h, w = RandomResizedCrop.get_params(
                        img, self.crop_range, self.crop_ratio
                    )
                    p1 = P1.copy()
                    p2 = P2.copy()
                    p2 = np.round(
                        np.multiply(
                            p2 - [0, i, j],
                            [1.0, self.crop_size[0] / h, self.crop_size[1] / w],
                        )
                    ).astype(np.int64)

                    valid_indexes_0 = np.logical_and(
                        p2[:, 1] < self.crop_size[0], p2[:, 1] >= 0
                    )
                    valid_indexes_1 = np.logical_and(
                        p2[:, 2] < self.crop_size[1], p2[:, 2] >= 0
                    )
                    valid_indexes = np.logical_and(valid_indexes_0, valid_indexes_1)
                    sum_indexes = valid_indexes.sum()
                    len_indexes = len(valid_indexes)
                    if sum_indexes > 1024 or sum_indexes / len_indexes > 0.75:
                        successfull = True
                imgs[id] = resized_crop(
                    img, i, j, h, w, self.crop_size, self.img_interpolation
                )
                if superpixels is not None:
                    sps[id] = resized_crop(
                        superpixels[id],
                        i,
                        j,
                        h,
                        w,
                        self.crop_size,
                        InterpolationMode.NEAREST,
                    )
                pairing_points_out = np.concatenate(
                    (pairing_points_out, p1[valid_indexes])
                )
                pairing_images_out = np.concatenate(
                    (pairing_images_out, p2[valid_indexes])
                )
        if superpixels is None:
            return pc, features, imgs, pairing_points_out, pairing_images_out, superpixels
        return pc, features, imgs, pairing_points_out, pairing_images_out, sps


class FlipHorizontal:
    """
    Flip horizontaly the image with probability p and adapt the matching accordingly.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pc, features, images, pairing_points, pairing_images, superpixels):
        w = images.shape[3]
        for i, img in enumerate(images):
            if random.random() < self.p:
                images[i] = hflip(img)
                if superpixels is not None:
                    superpixels[i] = hflip(superpixels[i: i + 1])
                mask = pairing_images[:, 0] == i
                pairing_images[mask, 2] = w - 1 - pairing_images[mask, 2]

        return pc, features, images, pairing_points, pairing_images, superpixels


class DropCuboids:
    """
    Drop random cuboids in a cloud
    """

    def __call__(self, pc, features, images, pairing_points, pairing_images, superpixels):
        range_xyz = torch.max(pc, axis=0)[0] - torch.min(pc, axis=0)[0]

        crop_range = np.random.random() * 0.2
        new_range = range_xyz * crop_range / 2.0

        sample_center = pc[np.random.choice(len(pc))]

        max_xyz = sample_center + new_range
        min_xyz = sample_center - new_range

        upper_idx = torch.sum((pc[:, 0:3] < max_xyz).to(torch.int32), 1) == 3
        lower_idx = torch.sum((pc[:, 0:3] > min_xyz).to(torch.int32), 1) == 3

        new_pointidx = ~((upper_idx) & (lower_idx))
        pc_out = pc[new_pointidx]
        features_out = features[new_pointidx]

        mask = new_pointidx[pairing_points]
        cs = torch.cumsum(new_pointidx, 0) - 1
        pairing_points_out = pairing_points[mask]
        pairing_points_out = cs[pairing_points_out]
        pairing_images_out = pairing_images[mask]

        successfull = True
        for id in range(len(images)):
            if np.sum(pairing_images_out[:, 0] == id) < 1024:
                successfull = False
        if successfull:
            return (
                pc_out,
                features_out,
                images,
                np.array(pairing_points_out),
                np.array(pairing_images_out),
                superpixels,
            )
        return pc, features, images, pairing_points, pairing_images, superpixels


def make_transforms_asymmetrical(config):
    """
    Read the config file and return the desired mixed transformation.
    """
    transforms = []
    if config["transforms_mixed"] is not None:
        for t in config["transforms_mixed"]:
            if t.lower() == "resizedcrop":
                transforms.append(
                    ResizedCrop(
                        image_crop_size=config["crop_size"],
                        image_crop_ratio=config["crop_ratio"],
                    )
                )
            elif t.lower() == "fliphorizontal":
                transforms.append(FlipHorizontal())
            elif t.lower() == "dropcuboids":
                transforms.append(DropCuboids())
            else:
                raise Exception(f"Unknown transformation {t}")
    if not len(transforms):
        return None
    return ComposeAsymmetrical(transforms)


def make_transforms_asymmetrical_val(config):
    """
    Read the config file and return the desired mixed transformation
    for the validation only.
    """
    transforms = []
    if config["transforms_mixed"] is not None:
        for t in config["transforms_mixed"]:
            if t.lower() == "resizedcrop":
                transforms.append(
                    ResizedCrop(image_crop_size=config["crop_size"], crop_center=True)
                )
    if not len(transforms):
        return None
    return ComposeAsymmetrical(transforms)
