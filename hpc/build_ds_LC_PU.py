from reco.datasetLCPU import LCGraphPU


data_root = "/mnt/ceph/users/ecuba/processed"
ds_name = "CloseByGamma200PUFull"
raw_dir = f"/mnt/ceph/users/ecuba/{ds_name}"

radius = 10
threshold = 0.2

sizes = [10, 100, 250, None]

for s in sizes:
    ds = LCGraphPU(
        ds_name + ".2",
        data_root,
        raw_dir,
        N_FILES=s,
        radius=radius,
        score_threshold=threshold,
    )
    del ds
