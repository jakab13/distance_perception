import pathlib
import os
import json
import mne
import numpy as np
from meegkit.dss import dss0, dss1
from meegkit.utils.covariances import tscov


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


def compute_transformation(epochs, conditions, keep, type="linear"):
    X = epochs.get_data().transpose(2, 1, 0)
    events = epochs.events
    idxs = [np.where(events[:, 2] == condition)[0] for condition in conditions]  # index of events per condition
    to_jd1, from_jd1, _, pwr = dss1(X, keep1=keep)  # compute the JD transformations
    del X
    Y = apply_transform(epochs.get_data(), to_jd1)  # apply the unmixing matrix to get the components
    if type == "diff":
        D = Y[idxs[-1], :, :].mean(axis=0) - Y[idxs[0], :, :].mean(axis=0)  # take the difference between two conditions
        D = D.T
    elif type == "linear":
        D = np.asarray([Y[idx, :, :].mean(axis=0) for idx in idxs])  # linear combination of conditions
    Y = Y.T  # shape must be in shape (n_times, n_chans[, n_trials])
    c0, nc0 = tscov(Y)
    c1, nc1 = tscov(D)
    c0 /= nc0  # divide by total weight to normalize
    c1 /= nc1
    to_jd2, from_jd2, _, pwr = dss0(c0, c1, keep1=keep)  # compute the transformations
    return to_jd1, from_jd1, to_jd2, from_jd2, pwr


def get_epochs(subject_id, experiment="vocal_effort"):
    DIR = pathlib.Path(os.getcwd())
    experiment_DIR = DIR / "analysis" / "data" / experiment
    epochs_folder = experiment_DIR / subject_id / "epochs"
    epochs = mne.read_epochs(epochs_folder / pathlib.Path(id + '-epo.fif'))
    if experiment == "pinknoise":
        epochs.apply_baseline(baseline=(-0.2, 0))
        epochs.shift_time(-0.1, relative=True)
    return epochs


def transform_epochs(epochs_from, epochs_to, conditions, keep, type="linear"):
    to_jd1, from_jd1, to_jd2, from_jd2, pwr = compute_transformation(epochs_from, conditions, keep, type=type)
    Y = apply_transform(epochs_to.get_data(), [to_jd1, to_jd2, from_jd2, from_jd1])
    transformed_epochs = epochs_to.copy()
    transformed_epochs._data = Y
    return transformed_epochs, to_jd1, from_jd1, to_jd2, from_jd2, pwr


def evoked_from_epochs(epochs):
    evoked = [epochs[event_id].average() for event_id in epochs.event_id
              if (event_id != "deviant") and (event_id != "button_press")]
    return evoked


experiment = "pinknoise"  # either "pinknoise" or "vocal_effort"
DIR = pathlib.Path(os.getcwd())
VE_DIR = DIR / "analysis" / "data" / "vocal_effort"  # pilot_laughter or pilot_noise
PN_DIR = DIR / "analysis" / "data" / "pinknoise"
with open(DIR / "analysis" / "preproc_config.json") as file:
    cfg = json.load(file)
montage_path = DIR / "analysis" / cfg["montage"]["name"]
montage = mne.channels.read_custom_montage(fname=montage_path)
ids_VE = list(name for name in os.listdir(VE_DIR) if os.path.isdir(os.path.join(VE_DIR, name)))
ids_PN = list(name for name in os.listdir(PN_DIR) if os.path.isdir(os.path.join(PN_DIR, name)))
ids = list(set(ids_VE) & set(ids_PN))
condition_ids = [1, 2, 3, 4, 5]
keep = 16
evokeds, evokeds_avrgd = cfg["epochs"][f"event_id_{experiment}"].copy(), \
                         cfg["epochs"][f"event_id_{experiment}"].copy()
for key in cfg["epochs"][f"event_id_{experiment}"]:
    evokeds[key], evokeds_avrgd[key] = list(), list()
channel_weights_all = []

for id in ids:
    epochs_from = get_epochs(id, "vocal_effort")
    epochs_to = get_epochs(id, "pinknoise")
    epochs_jd, to_jd1, from_jd1, to_jd2, from_jd2, pwr = transform_epochs(epochs_from, epochs_to, condition_ids, keep)

    evoked_jd = evoked_from_epochs(epochs_jd)
    evoked_from = evoked_from_epochs(epochs_from)
    evoked_to = evoked_from_epochs(epochs_to)
    evoked_diff = [mne.combine_evoked([evoked_jd[i-1], evoked_to[i-1]], weights=[1, -1]) for i in condition_ids]

    for condition in evoked_jd:
        evokeds[condition.comment].append(condition)
        if len(evokeds[condition.comment]) == len(ids):
            evokeds_avrgd[condition.comment] = mne.grand_average(
                evokeds[condition.comment])
    channel_weights = from_jd2 @ from_jd1
    channel_weights_all.append(channel_weights)

mne.viz.plot_compare_evokeds(evoked_from, picks="FCz", title="JD from")
mne.viz.plot_compare_evokeds(evoked_to, picks="FCz", title="JD to")
mne.viz.plot_compare_evokeds(evoked_jd, picks="FCz", title="Transformed")
mne.viz.plot_compare_evokeds(evoked_diff, picks="FCz", title="Diff")

mne.viz.plot_compare_evokeds(ignore_conds(evokeds_avrgd, "deviant", "button_press"), picks="FCz")

channel_weights_avrgd = np.asarray(channel_weights_all).mean(axis=0)
