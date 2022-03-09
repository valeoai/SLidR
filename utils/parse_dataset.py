import re
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


if __name__ == "__main__":
    """
    File used to generate the parametrizing splits
    """
    nusc = NuScenes(
        version="v1.0-trainval", dataroot="datasets/nuscenes/", verbose=True
    )
    phase_scenes = create_splits_scenes()["train"]
    n_rain = 0
    n_night = 0
    n_singapore = 0
    total = 0
    for scene_idx in range(len(nusc.scene)):
        scene = nusc.scene[scene_idx]
        if scene["name"] in phase_scenes:
            description = re.split("[, ]", scene["description"].lower())
            rain = "rain" in description
            night = "night" in description
            singapore = nusc.get("log", scene["log_token"])["location"].startswith(
                "singapore"
            )
            n_rain += rain
            n_night += night
            n_singapore += singapore
            total += 1

    print(
        f"Statistics in the train set:\n"
        f"{total} scenes\n"
        f"{n_rain} raining scenes\n"
        f"{n_night} night-time scenes\n"
        f"{n_singapore} scenes in Singapore\n"
        f"{total - n_singapore} scenes in Boston"
    )

    phase_scenes = create_splits_scenes()["val"]
    n_rain = 0
    n_night = 0
    n_singapore = 0
    total = 0
    for scene_idx in range(len(nusc.scene)):
        scene = nusc.scene[scene_idx]
        if scene["name"] in phase_scenes:
            description = re.split("[, ]", scene["description"].lower())
            rain = "rain" in description
            night = "night" in description
            singapore = nusc.get("log", scene["log_token"])["location"].startswith(
                "singapore"
            )
            n_rain += rain
            n_night += night
            n_singapore += singapore
            total += 1

    print(
        f"Statistics in the val set:\n"
        f"{total} scenes\n"
        f"{n_rain} raining scenes\n"
        f"{n_night} night-time scenes\n"
        f"{n_singapore} scenes in Singapore\n"
        f"{total - n_singapore} scenes in Boston"
    )

    while True:
        verifying = [
            "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032",
            "scene-0042", "scene-0045", "scene-0049", "scene-0052", "scene-0054",
            "scene-0056", "scene-0066", "scene-0067", "scene-0073", "scene-0131",
            "scene-0152", "scene-0166", "scene-0168", "scene-0183", "scene-0190",
            "scene-0194", "scene-0208", "scene-0210", "scene-0211", "scene-0241",
            "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
            "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306",
            "scene-0350", "scene-0352", "scene-0358", "scene-0361", "scene-0365",
            "scene-0368", "scene-0377", "scene-0388", "scene-0391", "scene-0395",
            "scene-0413", "scene-0427", "scene-0428", "scene-0438", "scene-0444",
            "scene-0452", "scene-0453", "scene-0459", "scene-0463", "scene-0464",
            "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
            "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658",
            "scene-0669", "scene-0678", "scene-0687", "scene-0701", "scene-0703",
            "scene-0706", "scene-0710", "scene-0715", "scene-0726", "scene-0735",
            "scene-0740", "scene-0758", "scene-0786", "scene-0790", "scene-0804",
            "scene-0806", "scene-0847", "scene-0856", "scene-0868", "scene-0882",
            "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
            "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024",
            "scene-1044", "scene-1058", "scene-1094", "scene-1098", "scene-1107",
        ]  # Chosen mini-val subset. Replace by a random generator to create another subset
        n_rain = 0
        n_night = 0
        n_singapore = 0
        total = 0
        for scene_idx in range(len(nusc.scene)):
            scene = nusc.scene[scene_idx]
            if scene["name"] in verifying:
                description = re.split("[, ]", scene["description"].lower())
                rain = "rain" in description
                night = "night" in description
                singapore = nusc.get("log", scene["log_token"])["location"].startswith(
                    "singapore"
                )
                n_rain += rain
                n_night += night
                n_singapore += singapore
        if n_singapore == 44 and n_rain == 20 and n_night == 12:
            break

    print(verifying)
