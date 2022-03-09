import torch
import argparse
from downstream.evaluate import evaluate
from utils.read_config import generate_config
from downstream.model_builder import make_model
from downstream.dataloader_kitti import make_data_loader as make_data_loader_kitti
from downstream.dataloader_nuscenes import make_data_loader as make_data_loader_nuscenes


def main():
    """
    Code for launching the downstream evaluation
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="specify the config for training"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Choose between nuScenes and KITTI"
    )
    args = parser.parse_args()
    if args.cfg_file is None and args.dataset is not None:
        if args.dataset.lower() == "kitti":
            args.cfg_file = "config/semseg_kitti.yaml"
        elif args.dataset.lower() == "nuscenes":
            args.cfg_file = "config/semseg_nuscenes.yaml"
        else:
            raise Exception(f"Dataset not recognized: {args.dataset}")
    elif args.cfg_file is None:
        args.cfg_file = "config/semseg_nuscenes.yaml"

    config = generate_config(args.cfg_file)
    if args.resume_path:
        config['resume_path'] = args.resume_path

    print("\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items()))))
    print("Creating the loaders")
    if config["dataset"].lower() == "nuscenes":
        phase = "verifying" if config['training'] in ("parametrize", "parametrizing") else "val"
        val_dataloader = make_data_loader_nuscenes(
            config, phase, num_threads=config["num_threads"]
        )
    elif config["dataset"].lower() == "kitti":
        val_dataloader = make_data_loader_kitti(
            config, "val", num_threads=config["num_threads"]
        )
    else:
        raise Exception(f"Dataset not recognized: {args.dataset}")
    print("Creating the model")
    model = make_model(config, config["pretraining_path"]).to(0)
    checkpoint = torch.load(config["resume_path"], map_location=torch.device(0))
    if "config" in checkpoint:
        for cfg in ("voxel_size", "cylindrical_coordinates"):
            assert checkpoint["config"][cfg] == config[cfg], (
                f"{cfg} is not consistant.\n"
                f"Checkpoint: {checkpoint['config'][cfg]}\n"
                f"Config: {config[cfg]}."
            )
    try:
        model.load_state_dict(checkpoint["model_points"])
    except KeyError:
        weights = {
            k.replace("model.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(weights)
    evaluate(model, val_dataloader, config)


if __name__ == "__main__":
    main()
