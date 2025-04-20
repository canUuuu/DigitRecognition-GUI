import h5py
with h5py.File('MODEL/CapsuleModel.h5', 'r') as file:
    print(list(file.keys()))
