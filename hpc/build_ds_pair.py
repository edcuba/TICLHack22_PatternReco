from reco.datasetPU import TracksterPairs

data_root = "/mnt/ceph/users/ecuba/processed"
ds_name = "MultiParticle"
raw_dir = f"/mnt/ceph/users/ecuba/{ds_name}"

sizes = [100, 500, 1000, None]

for s in sizes:
    ds = TracksterPairs(
        ds_name,
        data_root,
        raw_dir,
        N_FILES=s,
        radius=10,
        bigT_e_th=30,
    )
    del ds
