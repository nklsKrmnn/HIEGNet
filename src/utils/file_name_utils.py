import re

def get_glom_index(file_name: str) -> int:
    match = re.search(r"_i(\d+).", file_name)
    if match:
        return int(match.group(1))
    else:
        return -1