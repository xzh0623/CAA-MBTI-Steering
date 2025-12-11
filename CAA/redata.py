import os
import glob
import json

path = os.getcwd()
folder_path = os.path.join(path, "datasets", "raw", "caa_datasets_16")
files = glob.glob(os.path.join(folder_path, 'caa_data_*.json'))

for file in files:
    data_list = []
    with open(file, 'r') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))

    with open(file, 'w') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
