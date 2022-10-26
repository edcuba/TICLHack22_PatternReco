from reco.dataset import TracksterGraph
from reco.distance import get_z_map

data_root = "/mnt/ceph/users/ecuba/processed"
ds_name = "MultiParticle"
raw_dir = f"/mnt/ceph/users/ecuba/{ds_name}"

z_map = get_z_map(data_root)

graph_feat = False
max_dist = 15
eng_thresh = 15

ds = TracksterGraph(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=5,
    MAX_DISTANCE=max_dist,
    ENERGY_THRESHOLD=eng_thresh,
    include_graph_features=graph_feat,
)

ds = TracksterGraph(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=10,
    MAX_DISTANCE=max_dist,
    ENERGY_THRESHOLD=eng_thresh,
    include_graph_features=graph_feat,
)

ds = TracksterGraph(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=15,
    MAX_DISTANCE=max_dist,
    ENERGY_THRESHOLD=eng_thresh,
    include_graph_features=graph_feat,
)

ds = TracksterGraph(
    ds_name,
    data_root,
    raw_dir,
    MAX_DISTANCE=max_dist,
    ENERGY_THRESHOLD=eng_thresh,
    include_graph_features=graph_feat,
)