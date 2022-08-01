import os
import copy
import torch
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from spconv.pytorch.utils import PointToVoxel
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud


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


def mean_vfe(voxel_features, voxel_num_points):
    # voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
    points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
    normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
    points_mean = points_mean / normalizer
    voxel_features = points_mean.contiguous()

    return voxel_features


def spconv_collate_pair_fn(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    (
        pc,
        coords,
        feats,
        images,
        pairing_points,
        pairing_images,
        num_points,
        superpixels,
    ) = list(zip(*list_data))
    batch_n_points, batch_n_pairings = [], []

    pc_batch = []
    offset = 0
    for batch_id in range(len(pc)):
        pc_batch.append(torch.cat((torch.ones((pc[batch_id].shape[0], 1)) * batch_id, pc[batch_id]), 1))
        pairing_points[batch_id][:] += offset
        offset += pc[batch_id].shape[0]

    offset = 0
    for batch_id in range(len(coords)):

        # Move batchids to the beginning
        coords[batch_id][:, 0] = batch_id
        pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]

        batch_n_points.append(coords[batch_id].shape[0])
        batch_n_pairings.append(pairing_points[batch_id].shape[0])
        offset += coords[batch_id].shape[0]

    # Concatenate all lists
    coords_batch = torch.cat(coords, 0).int()
    pc_batch = torch.cat(pc_batch, 0)
    pairing_points = torch.tensor(np.concatenate(pairing_points))
    pairing_images = torch.tensor(np.concatenate(pairing_images))
    feats_batch = torch.cat(feats, 0).float()
    images_batch = torch.cat(images, 0).float()
    superpixels_batch = torch.tensor(np.concatenate(superpixels))
    num_points = torch.cat(num_points, 0)
    feats_batch = mean_vfe(feats_batch, num_points)
    return {
        "pc": pc_batch,
        "coordinates": coords_batch,
        "voxels": feats_batch,
        "input_I": images_batch,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        "batch_n_pairings": batch_n_pairings,
        "num_points": num_points,
        "superpixels": superpixels_batch,
    }


class NuScenesMatchDatasetSpconv(Dataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
        self,
        phase,
        config,
        shuffle=False,
        cloud_transforms=None,
        mixed_transforms=None,
        **kwargs,
    ):
        self.phase = phase
        self.shuffle = shuffle
        self.cloud_transforms = cloud_transforms
        self.mixed_transforms = mixed_transforms
        if config["dataset"] == "nuscenes":
            self.voxel_size = [0.1, 0.1, 0.2]  # nuScenes
            self.point_cloud_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], dtype=np.float32)  # nuScenes
            MAX_POINTS_PER_VOXEL = 10  # nuScenes
            MAX_NUMBER_OF_VOXELS = 60000  # nuScenes
            self._voxel_generator = PointToVoxel(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=4,
                max_num_points_per_voxel=MAX_POINTS_PER_VOXEL,
                max_num_voxels=MAX_NUMBER_OF_VOXELS
            )
        else:
            raise Exception("Dataset unknown")
        self.superpixels_type = config["superpixels_type"]
        self.bilinear_decoder = config["decoder"] == "bilinear"
        self.num_point_features = 4

        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot="datasets/nuscenes", verbose=False
            )

        self.list_keyframes = []
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
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
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

    def create_list_of_scans(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        list_data = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            list_data.append(current_sample["data"])
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        pc = pc_original.points
        dist = pc[0] * pc[0] + pc[1] * pc[1]
        mask = (dist <= 2621.44) & \
               (pc[2] >= self.point_cloud_range[2]) & \
               (pc[2] <= self.point_cloud_range[5])
        pc_original = LidarPointCloud(pc[:, mask])

        pc_ref = pc_original.points

        images = []
        superpixels = []
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        if self.shuffle:
            np.random.shuffle(camera_list)
        for i, camera_name in enumerate(camera_list):
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
            sp = Image.open(
                f"superpixels/nuscenes/"
                f"superpixels_{self.superpixels_type}/{cam['token']}.png"
            )
            superpixels.append(np.array(sp))

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < im.shape[1] - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < im.shape[0] - 1)
            matching_points = np.where(mask)[0]
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64)
            images.append(im / 255)
            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * i,
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )
        return pc_ref.T, images, pairing_points, pairing_images, np.stack(superpixels)

    def __len__(self):
        return len(self.list_keyframes)

    def _voxelize(self, points):
        voxel_output = self._voxel_generator.generate_voxel_with_id(points)
        voxels, coordinates, num_points, indexes = voxel_output
        return voxels, coordinates, num_points

    def __getitem__(self, idx):
        (
            pc,
            images,
            pairing_points,
            pairing_images,
            superpixels,
        ) = self.map_pointcloud_to_image(self.list_keyframes[idx])
        superpixels = torch.tensor(superpixels)

        intensity = torch.tensor(pc[:, 3:])
        pc = torch.tensor(pc[:, :3])
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))

        if self.cloud_transforms:
            pc = self.cloud_transforms(pc)
        if self.mixed_transforms:
            (
                pc,
                intensity,
                images,
                pairing_points,
                pairing_images,
                superpixels,
            ) = self.mixed_transforms(
                pc, intensity, images, pairing_points, pairing_images, superpixels
            )

        pc = torch.cat((pc, intensity), 1)

        voxels, coordinates, num_points = self._voxelize(pc)

        discrete_coords = torch.cat(
            (
                torch.zeros(coordinates.shape[0], 1, dtype=torch.int32),
                coordinates,
            ),
            1,
        )
        voxels = voxels
        num_points = num_points

        return (
            pc,
            discrete_coords,
            voxels,
            images,
            pairing_points,
            pairing_images,
            num_points,
            superpixels,
        )
