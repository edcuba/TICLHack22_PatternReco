import numpy as np
from .plotting import plot_fractions_hist


def get_eid_splits(tracksters, simtracksters, associations, match_threshold=0.2):
    perfect_eids = []
    split_eids = []

    for eid in range(len(tracksters["event"].array())):

        # get the number of tracksters
        num_rec_t = tracksters["NTracksters"].array()[eid]
        num_sim_t = simtracksters["stsSC_NTracksters"].array()[eid]

        # get reco <-> sim maps
        r2s = np.array(associations["tsCLUE3D_recoToSim_SC_score"].array()[eid])
        s2r = np.array(associations["tsCLUE3D_simToReco_SC_score"].array()[eid])

        if num_rec_t == num_sim_t:  # matching number of tracksters
            perf_match = True  # assume perfect match

            for ti, reco_t in enumerate(r2s):
                sim_mask = (reco_t < match_threshold).astype(int)
                if np.sum(sim_mask) != 1:
                    perf_match = False
                    # print(f"Event {eid}/{ti} reco to sim mismatch: {reco_t}")

            for ti, sim_t in enumerate(s2r):
                reco_mask = (sim_t < match_threshold).astype(int)
                if np.sum(reco_mask) != 1:
                    perf_match = False
                    # print(f"Event {eid}/{ti} sim to reco mismatch: {sim_t}")

            if perf_match:
                perfect_eids.append(eid)

        elif num_rec_t > num_sim_t: # split
            split_eids.append(eid)
        else:                       # overmerged
            # print(f"Event {eid} is overmerged ({num_sim_t} in {num_rec_t})")
            pass

    return perfect_eids, split_eids


def get_highest_energy_fraction_simtracksters(tracksters, simtracksters, associations, eid):
    num_rec_t = tracksters["NTracksters"].array()[eid]

    # get the raw energy of reco and sim tracksters
    # raw_energy = np.array(tracksters["raw_energy"].array()[split_eid])
    st_raw_energy = np.array(simtracksters["stsSC_raw_energy"].array()[eid])

    # get the shared energy mapping
    s2ri = np.array(associations["tsCLUE3D_simToReco_SC"].array()[eid])
    s2r_SE = np.array(associations["tsCLUE3D_simToReco_SC_sharedE"].array()[eid])

    # keep the highest fraction
    reco_fr = [0] * num_rec_t
    reco_st = [-1] * num_rec_t

    # for each trackster, get the simtrackster with the highest energy fraction
    for st_i, reco_indexes, shared_energies in zip(range(len(s2ri)), s2ri, s2r_SE):
        st_e = st_raw_energy[st_i]
        # print(f"Event {split_eid} Sim trackster {st_i} with energy {st_e:.4f}:")
        for rt_i, sh_e in zip(reco_indexes, shared_energies):
            fraction = sh_e / st_e
            # rt_e = raw_energy[rt_i]
            # print(f"\tshared energy with {rt_i} ({rt_e:.4f}): {sh_e:.4f} / {st_e:.4f} = {fraction:.4f}")
            if fraction > reco_fr[rt_i]:
                reco_fr[rt_i] = fraction
                reco_st[rt_i] = st_i

    return reco_fr, reco_st



def split_on_shared_energy(tracksters, simtracksters, h_frac, event_eids, complete_threshold=0.5, histogram=False):
    """
    Compute complete and incomplete tracksters based on shared energy
    """
    all_fractions = []
    incomplete_fractions = []
    complete_fractions = []
    num_tracksters = 0
    num_complete = 0
    num_incomplete = 0

    complete_tracksters = []
    incomplete_tracksters = []
    for split_eid in event_eids:
        # get the number of extra tracksters
        num_rec_t = tracksters["NTracksters"].array()[split_eid]
        num_sim_t = simtracksters["stsSC_NTracksters"].array()[split_eid]
        num_extra_t = num_rec_t - num_sim_t

        # get highest energy fraction simtracksters
        reco_fr, _ = h_frac[split_eid]

        # sort the tracksters by the energy fraction
        sorted_t_by_fr = sorted(range(len(reco_fr)), key=lambda x: reco_fr[x])
        incomplete_tracksters.append((split_eid, sorted_t_by_fr[:num_extra_t]))

        # use the remaining complete tracksters
        for i, t in enumerate(sorted_t_by_fr[num_extra_t:]):
            if reco_fr[t] > complete_threshold:
                # following tracksters have higher energies
                complete_t = sorted_t_by_fr[i+num_extra_t:]
                complete_tracksters.append((split_eid, complete_t))
                num_complete += len(complete_t)
                break

        # save fraction stats
        num_tracksters += num_rec_t
        num_incomplete += num_extra_t
        all_fractions += reco_fr

        sorted_fr_ = sorted(reco_fr)
        incomplete_fractions += sorted_fr_[:num_extra_t]
        complete_fractions += sorted_fr_[num_extra_t:]

    if histogram:
        plot_fractions_hist(all_fractions, complete_fractions, incomplete_fractions)

    print(f"Total reco tracksters: {num_tracksters}, complete: {num_complete}, Incomplete: {num_incomplete}")
    return complete_tracksters, incomplete_tracksters


def unfold_tracksters(tracksters, eids):
    return [
        (eid, list(range(tracksters["NTracksters"].array()[eid])))
        for eid in eids
    ]


def get_pairs(incomplete_tuples, associations, h_frac):
    pairs = []
    w_itself = 0
    w_none = 0

    for eid, idxs in incomplete_tuples:
        r2si = associations["tsCLUE3D_recoToSim_SC"].array()[eid]
        r2s = associations["tsCLUE3D_recoToSim_SC_score"].array()[eid]

        # id of simtrackster it should merge with
        reco_fr, reco_st = h_frac[eid]

        for idx in idxs:
            match = np.argmin(r2s[idx])
            sidx = r2si[idx][match]

            matches = np.where(np.array(reco_st) == sidx)[0]
            candidates = sorted(matches, key=lambda x: reco_fr[x], reverse=True)

            if not candidates:
                # no candidates to merge with
                w_none += 1
                continue

            if candidates[0] == idx:
                # merging with itself?
                w_itself += 1
                continue

            pairs.append((eid, idx, candidates[0], 1))

            unmatches = np.where(np.array(reco_st) != sidx)[0]
            for unmatch in unmatches:
                # add up to 1 unmatch
                pairs.append((eid, idx, unmatch, 0))
                break

    print("Pairs:", len(pairs), "No candidates:", w_none, "Merging with itself:", w_itself)
    return pairs