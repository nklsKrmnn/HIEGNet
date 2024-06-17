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