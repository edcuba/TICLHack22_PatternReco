from reco.dataset import TracksterPairs


data_root = "/mnt/ceph/users/ecuba/processed"
ds_name = "CloseByTwoPion"
raw_dir = f"/mnt/ceph/users/ecuba/{ds_name}"

max_dist = 20


ds = TracksterPairs(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=100,
    MAX_DISTANCE=max_dist,
    ENERGY_THRESHOLD=10,
)

ds = TracksterPairs(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=200,
    MAX_DISTANCE=max_dist,
    ENERGY_THRESHOLD=10,
)

ds = TracksterPairs(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=300,
    MAX_DISTANCE=max_dist,
    ENERGY_THRESHOLD=10,
)

ds = TracksterPairs(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=400,
    MAX_DISTANCE=max_dist,
    ENERGY_THRESHOLD=10,
)

ds = TracksterPairs(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=500,
    MAX_DISTANCE=max_dist,
    ENERGY_THRESHOLD=10,
)
