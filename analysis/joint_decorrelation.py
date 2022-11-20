import pathlib
import os
import json
import mne
import numpy as np
from meegkit.dss import dss0, dss1
from meegkit.utils.covariances import tscov
import matplotlib.pyplot as plt

def ignore_conds(d, *keys):
    return dict(filter(lambda key_value: key_value[0] not in keys, d.items()))

def apply_transform(data, transforms):
    if not isinstance(transforms, list):
        transforms = [transforms]
    n_epochs, n_channels, n_times = data.shape
    data = data.transpose(1, 0, 2)
    data = data.reshape(n_channels, n_epochs * n_times).T
    for i, transform in enumerate(transforms):
        if i == 0:
            transformed = data @ transform
        else:
            transformed = transformed @ transform
    transformed = np.reshape(transformed.T, [-1, n_epochs, n_times]).transpose([1, 0, 2])
    return transformed

def compute_transformation(epochs, conditions, keep):
    X = epochs.get_data().transpose(2, 1, 0)
    events = epochs.events
    idxs = [np.where(events[:, 2] == condition)[0] for condition in conditions]
    to_jd1, from_jd1, _, pwr = dss1(X, keep1=keep)  # compute the transformations
    del X
    Y = apply_transform(epochs.get_data(), to_jd1)  # apply the unmixing matrix to get the components
    D_diff = Y[idxs[4], :, :].mean(axis=0) - Y[idxs[1], :, :].mean(axis=0)
    D_linear = np.asarray([Y[idx, :, :].mean(axis=0) for idx in idxs])
    Y, D_diff = Y.T, D_diff.T  # shape must be in shape (n_times, n_chans[, n_trials])
    c0, nc0 = tscov(Y)
    c1_linear, nc1_linear = tscov(D_linear)
    c1_diff, nc1_diff = tscov(D_diff)
    c0 /= nc0  # divide by total weight to normalize
    c1_linear /= nc1_linear
    c1_diff /= nc1_diff
    to_jd2_linear, from_jd2_linear, _, pwr = dss0(c0, c1_linear, keep1=keep)  # compute the transformations
    to_jd2_diff, from_jd2_diff, _, pwr = dss0(c0, c1_diff, keep1=keep)
    return to_jd1, from_jd1, to_jd2_linear, from_jd2_linear

experiment = "pinknoise"  # either "pinknoise" or "vocal_effort"
# get pilot folder directory.
DIR = pathlib.Path(os.getcwd())
VE_DIR = DIR / "analysis" / "data" / "vocal_effort"  # pilot_laughter or pilot_noise
PN_DIR = DIR / "analysis" / "data" / "pinknoise"
with open(DIR / "analysis" / "preproc_config.json") as file:
    cfg = json.load(file)

montage_path = DIR / "analysis" / cfg["montage"]["name"]
montage = mne.channels.read_custom_montage(fname=montage_path)

# get subject ids
ids_VE = list(name for name in os.listdir(VE_DIR) if os.path.isdir(os.path.join(VE_DIR, name)))
ids_PN = list(name for name in os.listdir(PN_DIR) if os.path.isdir(os.path.join(PN_DIR, name)))
ids = list(set(ids_VE) & set(ids_PN))
condition_ids = [1, 2, 3, 4, 5]
keep = 10
# initialise evokeds and related objects
evokeds, evokeds_avrgd = cfg["epochs"][f"event_id_{experiment}"].copy(), \
                         cfg["epochs"][f"event_id_{experiment}"].copy()
for key in cfg["epochs"][f"event_id_{experiment}"]:
    evokeds[key], evokeds_avrgd[key] = list(), list()

channel_weights_all = []

for id in ids:
    epochs_folder_VE = VE_DIR / id / "epochs"
    epochs_VE = mne.read_epochs(epochs_folder_VE / pathlib.Path(id + '-epo.fif'))
    epochs_folder_PN = PN_DIR / id / "epochs"
    epochs_PN = mne.read_epochs(epochs_folder_PN / pathlib.Path(id + '-epo.fif'))
    epochs_PN.apply_baseline(baseline=(-0.2, 0))
    epochs_PN.shift_time(-0.1, relative=True)

    to_jd1, from_jd1, to_jd2, from_jd2 = compute_transformation(epochs_VE, condition_ids, keep)
    Y = apply_transform(epochs_PN.get_data(), [to_jd1, to_jd2, from_jd2, from_jd1])
    channel_weights = from_jd2 @ from_jd1
    channel_weights_all.append(channel_weights)
    epochs_jd = epochs_PN.copy()
    epochs_jd._data = Y
    evoked = [epochs_jd[condition_id].average() for condition_id in cfg["epochs"][f"event_id_{experiment}"].keys()]
    evoked_PN = [epochs_PN[condition_id].average() for condition_id in cfg["epochs"][f"event_id_{experiment}"].keys()]
    for condition in evoked:
        evokeds[condition.comment].append(condition)
        if len(evokeds[condition.comment]) == len(ids):
            evokeds_avrgd[condition.comment] = mne.grand_average(
                evokeds[condition.comment])

    mne.viz.plot_compare_evokeds(evoked[1:], picks="FCz")
    mne.viz.plot_compare_evokeds(evoked_PN[1:], picks="FCz")


mne.viz.plot_compare_evokeds(ignore_conds(evokeds_avrgd, "deviant", "button_press"), picks="FCz")

channel_weights_avrgd = np.asarray(channel_weights_all).mean(axis=0)
mne.viz.plot_topomap(channel_weights_avrgd[0:10, :].mean(axis=0), epochs_VE.info)
