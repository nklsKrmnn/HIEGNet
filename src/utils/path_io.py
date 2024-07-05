import os
import re

def get_glom_index(file_name: str) -> int:
    match = re.search(r"_i(\d+).", file_name)
    if match:
        return int(match.group(1))
    else:
        return -1

def get_patient(file_name: str) -> int:
    match = re.search(r"_p(\d+)_", file_name)
    if match:
        return int(match.group(1))
    else:
        return -1


def get_path_up_to(current_path, directory_name):
    # Split the path into its components
    path_components = current_path.split(os.sep)

    # Find the index of the target directory in the path components
    if directory_name in path_components:
        target_index = path_components.index(directory_name)
        # Join the path components up to and including the target directory
        truncated_path = os.sep.join(path_components[:target_index])
        return truncated_path
    else:
        raise ValueError(f"Directory '{directory_name}' not found in the current file path")