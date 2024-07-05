import pickle

path = "/home/dascim/repos/histograph/data/input/processed/test_data_filenames.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

new = []
new.append({"file_name": data[0]["filename"], "set": data[0]["set"]})
new.append({"file_name": data[1]["filename"], "set": data[0]["set"]})

with open(path, "wb") as f:
    pickle.dump(new, f)