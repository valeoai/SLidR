import os
import torch
import numpy as np
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from MinkowskiEngine.utils import sparse_quantize
from utils.transforms import make_transforms_clouds
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud


# parametrizing set, to try out different parameters
CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


def custom_collate_fn(list_data):
    """
    Custom collate function adapted for creating batches with MinkowskiEngine.
    """
    input = list(zip(*list_data))
    # whether the dataset returns labels
    labelized = len(input) == 6
    # evaluation_labels are per points, labels are per voxels
    if labelized:
        xyz, coords, feats, labels, evaluation_labels, inverse_indexes = input
    else:
        xyz, coords, feats, inverse_indexes = input

    coords_batch, len_batch = [], []

    # create a tensor of coordinates of the 3D points
    # note that in ME, batche index and point indexes are collated in the same dimension
    for batch_id, coo in enumerate(coords):
        N = coords[batch_id].shape[0]
        coords_batch.append(
            torch.cat((torch.ones(N, 1, dtype=torch.int32) * batch_id, coo), 1)
        )
        len_batch.append(N)

    # Collate all lists on their first dimension
    coords_batch = torch.cat(coords_batch, 0).int()
    feats_batch = torch.cat(feats, 0).float()
    if labelized:
        labels_batch = torch.cat(labels, 0).long()
        return {
            "pc": xyz,  # point cloud
            "sinput_C": coords_batch,  # discrete coordinates (ME)
            "sinput_F": feats_batch,  # point features (N, 3)
            "len_batch": len_batch,  # length of each batch
            "labels": labels_batch,  # labels for each (voxelized) point
            "evaluation_labels": evaluation_labels,  # labels for each point
            "inverse_indexes": inverse_indexes,  # labels for each point
        }
    else:
        return {
            "pc": xyz,
            "sinput_C": coords_batch,
            "sinput_F": feats_batch,
            "len_batch": len_batch,
            "inverse_indexes": inverse_indexes,
        }


class NuScenesDataset(Dataset):
    """
    Dataset returning a lidar scene and associated labels.
    """

    def __init__(self, phase, config, transforms=None, cached_nuscenes=None):
        self.phase = phase
        self.labels = self.phase != "test"
        self.transforms = transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        self.use_intensity = config["use_intensity"]

        if phase != "test":
            if cached_nuscenes is not None:
                self.nusc = cached_nuscenes
            else:
                self.nusc = NuScenes(
                    version="v1.0-trainval", dataroot="datasets/nuscenes", verbose=False
                )
        else:
            self.nusc = NuScenes(
                version="v1.0-test", dataroot="datasets/nuscenes", verbose=False
            )

        self.list_tokens = []

        # a skip ratio can be used to reduce the dataset size
        # and accelerate experiments
        if phase in ("val", "verifying"):
            skip_ratio = 1
        else:
            try:
                skip_ratio = config["dataset_skip_step"]
            except KeyError:
                skip_ratio = 1
        skip_counter = 0
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        # create a list of all keyframe scenes
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_tokens(scene)

        # labels' names lookup table
        self.eval_labels = {
            0: 0, 1: 0, 2: 7, 3: 7, 4: 7, 5: 0, 6: 7, 7: 0, 8: 0, 9: 1, 10: 0, 11: 0,
            12: 8, 13: 0, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0,
        }

    def create_list_of_tokens(self, scene):
        # Get first in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            next_sample_token = current_sample["next"]
            self.list_tokens.append(current_sample["data"]["LIDAR_TOP"])
            current_sample_token = next_sample_token

    def __len__(self):
        return len(self.list_tokens)

    def __getitem__(self, idx):
        lidar_token = self.list_tokens[idx]
        pointsensor = self.nusc.get("sample_data", lidar_token)
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        points = LidarPointCloud.from_file(pcl_path).points.T
        # get the points (4th coordinate is the point intensity)
        pc = points[:, :3]
        if self.labels:
            lidarseg_labels_filename = os.path.join(
                self.nusc.dataroot, self.nusc.get("lidarseg", lidar_token)["filename"]
            )
            points_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8)

        pc = torch.tensor(pc)

        # apply the transforms (augmentation)
        if self.transforms:
            pc = self.transforms(pc)

        if self.cylinder:
            # Transform to cylinder coordinate and scale for given voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            # corresponds to a split each 1°
            phi = torch.atan2(y, x) * 180 / np.pi
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization
        discrete_coords, indexes, inverse_indexes = sparse_quantize(
            coords_aug, return_index=True, return_inverse=True
        )

        # use those voxels features
        if self.use_intensity is True:
            unique_feats = torch.tensor(points[indexes][:, 3:])
        else:
            unique_feats = torch.ones((discrete_coords.shape[0], 1), dtype=torch.int32)

        if self.labels:
            points_labels = torch.tensor(
                np.vectorize(self.eval_labels.__getitem__)(points_labels),
                dtype=torch.int32,
            )
            unique_labels = points_labels[indexes]

        if self.labels:
            return (
                pc,
                discrete_coords,
                unique_feats,
                unique_labels,
                points_labels,
                inverse_indexes,
            )
        else:
            return pc, discrete_coords, unique_feats, inverse_indexes


def make_data_loader(config, phase, num_threads=0):
    """
    Create the data loader for a given phase and a number of threads.
    This function is not used with pytorch lightning, but is used when evaluating.
    """
    # select the desired transformations
    if phase == "train":
        transforms = make_transforms_clouds(config)
    else:
        transforms = None

    # instantiate the dataset
    dset = NuScenesDataset(phase=phase, transforms=transforms, config=config)
    collate_fn = custom_collate_fn
    batch_size = config["batch_size"] // config["num_gpus"]

    # create the loader
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=phase == "train",
        num_workers=num_threads,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=phase == "train",
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    )
    return loader
