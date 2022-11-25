from reco.datasetPU import TracksterPairsPU


data_root = "/mnt/ceph/users/ecuba/processed"
ds_name = "CloseByGamma200PUFull"
raw_dir = f"/mnt/ceph/users/ecuba/{ds_name}"

radius = 10
threshold = 0.2

ds = TracksterPairsPU(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=100,
    radius=radius,
    score_threshold=threshold,
)
del ds

ds = TracksterPairsPU(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=250,
    radius=radius,
    score_threshold=threshold,
)
del ds

ds = TracksterPairsPU(
    ds_name,
    data_root,
    raw_dir,
    radius=radius,
    score_threshold=threshold,
)
del ds