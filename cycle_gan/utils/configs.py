from typing import Dict, Any, Optional

import yaml

from pathlib import Path
from argparse import Namespace


def load_config_from_yaml(config_path: Path):
    """ Load config from yaml file

    Args:
        config_path (str): path to config file

    Returns:
        config (Namespace): config dictionary
    """
    with open(config_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    ns = dict_to_namespace(config_dict)
    return ns


def dict_to_namespace(dict_: Dict, vtype: Optional[Any] = None) -> Namespace:
    """ Converts a nested dictionary to a Namespace object recursively 
    
    Args:
        dict (dict): dictionary to convert
        vtype (type): type to cast values to
    
    Returns:
        Namespace: Namespace object
    """
    
    def recurse(_dict):
        # If all children are not dicts
        ret = {}
        for k, v in _dict.items():
            if isinstance(v, dict):
                ret[k] = recurse(v)
            else:
                if vtype is not None:
                    ret[k] = vtype(v)
                else:
                    ret[k] = v
        return Namespace(**ret)

    return recurse(dict_)
