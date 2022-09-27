import uproot
from os import walk

from reco.dataset import match_trackster_pairs

# configuration
data_path = "/Users/ecuba/data/multiparticle_complet/"
ENERGY_TH = 10      # energy threshold
MAX_DISTANCE = 10   # cm
DISTANCE_TYPE = "pairwise"
CONFIDENCE = 0.5

# get all input files
files = []
for (dirpath, dirnames, filenames) in walk(data_path):
    files.extend(filenames)
    break

print("Data files:", len(files))


for f in files[:1]:
    file_name = data_path + f
    tracksters = uproot.open({file_name: "ticlNtuplizer/tracksters"})
    simtracksters = uproot.open({file_name: "ticlNtuplizer/simtrackstersSC"})
    associations = uproot.open({file_name: "ticlNtuplizer/associations"})

    event_e_pairs = match_trackster_pairs(
        tracksters,
        simtracksters,
        associations,
        energy_threshold=ENERGY_TH,
        distance_threshold=MAX_DISTANCE,
        distance_type=DISTANCE_TYPE,
        confidence_threshold=CONFIDENCE
    )

    raw_event_e = tracksters["raw_energy"].array()

    for eid in range(len(raw_event_e)):
        raw_e = raw_event_e[eid]
        e_pairs = event_e_pairs[eid]

        new_idx_map = {}
        merge_map = {little: big for little, big, _ in e_pairs}

        new_tracksters = []

        for tr_id in range(len(raw_e)):
            # only keep the tracksters that are not going to be merged
            if tr_id in merge_map.keys():
                # small trackster, ignore
                continue

            # create the new entry
            new_tracksters.append([tr_id])
            new_idx_map[tr_id] = len(new_tracksters) - 1

        # now fill in the tracksters to be merged
        for little, big in merge_map.items():
            new_big_idx = new_idx_map[big]
            new_tracksters[new_big_idx].append(little)

        print(f"{f} | Event {eid}:\t{len(raw_e)}\t -> {len(new_tracksters)}")




