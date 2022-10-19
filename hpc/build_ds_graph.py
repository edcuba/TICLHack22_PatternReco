from reco.dataset import TracksterGraph


data_root = "/mnt/ceph/users/ecuba/processed"
ds_name = "CloseByTwoPion"
raw_dir = f"/mnt/ceph/users/ecuba/{ds_name}"

graph_feat = True


ds = TracksterGraph(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=100,
    MAX_DISTANCE=10,
    ENERGY_THRESHOLD=10,
    include_graph_features=graph_feat,
)

ds = TracksterGraph(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=200,
    MAX_DISTANCE=10,
    ENERGY_THRESHOLD=10,
    include_graph_features=graph_feat,
)

ds = TracksterGraph(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=300,
    MAX_DISTANCE=10,
    ENERGY_THRESHOLD=10,
    include_graph_features=graph_feat,
)

ds = TracksterGraph(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=400,
    MAX_DISTANCE=10,
    ENERGY_THRESHOLD=10,
    include_graph_features=graph_feat,
)

ds = TracksterGraph(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=500,
    MAX_DISTANCE=10,
    ENERGY_THRESHOLD=10,
    include_graph_features=graph_feat,
)
