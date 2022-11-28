from reco.datasetPU import TracksterGraphPU

data_root = "/mnt/ceph/users/ecuba/processed"
ds_name = "CloseByGamma200PUFull"
raw_dir = f"/mnt/ceph/users/ecuba/{ds_name}"

radius = 10
threshold = 0.2

sizes = [10, 50, 100, 250, None]

for s in sizes:
    ds = TracksterGraphPU(
        ds_name,
        data_root,
        raw_dir,
        N_FILES=s,
        radius=10,
    )
    del ds
