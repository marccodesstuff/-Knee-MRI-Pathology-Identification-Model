import h5py

# Replace 'file_path.h5' with the path to your .h5 file
file_path = 'dataset/file1000022.h5'

with h5py.File(file_path, 'r') as f:
    print("Keys in the H5 file:")
    print(list(f.keys()))
    for key in f.keys():
        print(f"Key: {key}, Shape: {f[key].shape}, Data Type: {f[key].dtype}")