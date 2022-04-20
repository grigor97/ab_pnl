import yaml


def load_cfg(yaml_file_path):
    """
    Loads a yaml config file
    :param yaml_file_path: path of yaml file
    :return: config corresponding the path
    """
    with open(yaml_file_path, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)

    return cfg
