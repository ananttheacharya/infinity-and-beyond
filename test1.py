import h5py
with h5py.File('Dataset/N-CMAPSS_DS03-012.h5', 'r') as f:
    print(list(f.keys()))
    for k in f.keys():
        print(k, f[k].shape if hasattr(f[k], 'shape') else 'group')