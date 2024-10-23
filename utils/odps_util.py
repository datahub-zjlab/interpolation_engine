from odps import ODPS
import json
import os


def find_root_via_git():
    current_path = os.path.abspath(__file__)
    while True:
        git_dir = os.path.join(current_path, '.git')
        if os.path.isdir(git_dir):
            return current_path
        else:
            current_path = os.path.dirname(current_path)
        if current_path == os.path.dirname(current_path):  # reached root
            break
    return current_path


def get_config():
    root_dir = find_root_via_git()
    with open(os.path.join(root_dir, "utils/config.json"), 'r') as file:
        config = json.load(file)
        return config['odps_auth']['access_id'], config['odps_auth']['secret_access_key'], config['odps_auth']['project'], config['odps_auth']['endpoint']

class Odps:
    def __init__(self):
        access_id, secret_access_key, project, endpoint = get_config()
        self.odps = ODPS(access_id=access_id, secret_access_key=secret_access_key, project=project, endpoint=endpoint)

    def get_odps_instance(self):
        return self.odps