import yaml
from datetime import datetime as dt


def generate_config(file):
    with open(file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        config["datetime"] = dt.today().strftime("%d%m%y-%H%M")

    return config
