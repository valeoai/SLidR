import torch
from tqdm import tqdm
from copy import deepcopy
from MinkowskiEngine import SparseTensor
from utils.metrics import compute_IoU


CLASSES_NUSCENES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

CLASSES_KITTI = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]


def evaluate(model, dataloader, config):
    """
    Function to evaluate the performances of a downstream training.
    It prints the per-class IoU, mIoU and fwIoU.
    """
    model.eval()
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []
        for batch in tqdm(dataloader):
            sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"], device=0)
            output_points = model(sparse_input).F
            if config["ignore_index"]:
                output_points[:, config["ignore_index"]] = -1e6

            torch.cuda.empty_cache()
            preds = output_points.argmax(1).cpu()
            offset = 0
            for j, lb in enumerate(batch["len_batch"]):
                inverse_indexes = batch["inverse_indexes"][j]
                predictions = preds[inverse_indexes + offset]

                # remove the ignored index entirely
                full_predictions.append(predictions)
                ground_truth.append(deepcopy(batch["evaluation_labels"][j]))
                offset += lb
            i += j
        m_IoU, fw_IoU, per_class_IoU = compute_IoU(
            torch.cat(full_predictions),
            torch.cat(ground_truth),
            config["model_n_out"],
            ignore_index=0,
        )
        print("Per class IoU:")
        if config["dataset"].lower() == "nuscenes":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_NUSCENES, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        elif config["dataset"].lower() == "kitti":
            print(
                *[
                    f"{a:20} - {b:.3f}"
                    for a, b in zip(CLASSES_KITTI, (per_class_IoU).numpy())
                ],
                sep="\n",
            )
        print()
        print(f"mIoU: {m_IoU}")
        print(f"fwIoU: {fw_IoU}")

    return m_IoU
