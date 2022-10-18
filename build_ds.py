from reco.dataset import TracksterGraph


data_root = "data"
ds_name = "CloseByTwoPion"
raw_dir = f"/Users/ecuba/data/{ds_name}"
file_name = f"{raw_dir}/new_ntuples_15101852_191.root"


ds = TracksterGraph(
    ds_name,
    data_root,
    raw_dir,
    N_FILES=100,
    MAX_DISTANCE=10,
    ENERGY_THRESHOLD=10,
    include_graph_features=False,
)