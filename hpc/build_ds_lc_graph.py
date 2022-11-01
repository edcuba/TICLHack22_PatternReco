from reco.dataset import PointCloudSet

data_root = "/mnt/ceph/users/ecuba/processed"
ds_name = "MultiParticle"
raw_dir = f"/mnt/ceph/users/ecuba/{ds_name}"

ds = PointCloudSet(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=10,
)

ds = PointCloudSet(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=20,
)

ds = PointCloudSet(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=50,
)

ds = PointCloudSet(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=100,
)

ds = PointCloudSet(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=200,
)

ds = PointCloudSet(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=500,
)

ds = PointCloudSet(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=1000,
)