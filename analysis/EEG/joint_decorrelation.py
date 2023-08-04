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
    to_jd1, from_jd1, pwr0, pwr1 = dss1(X)  # compute the JD transformations
    del X
    to_jd1 = to_jd1[:, np.argsort(pwr1)[::-1]]  # sort them by magnitude
    from_jd1 = from_jd1[np.argsort(pwr1)[::-1], :]
    to_jd1 = to_jd1[:, 0:keep]  # only keep the largest ones
    from_jd1 = from_jd1[0:keep, :]
    Y = apply_transform(epochs.get_data(), to_jd1)  # apply the unmixing matrix to get the components
    if type == "diff":
        D = Y[idxs[-1], :, :].mean(axis=0) - Y[idxs[0], :, :].mean(axis=0)  # take the difference between two conditions
        D = D.T
    elif type == "linear":
        # D = np.asarray([Y[idx, :, :].mean(axis=0) for idx in idxs])  # linear combination of conditions
        D = np.asarray([Y[idx, :, :].mean(axis=0) * (i-2) * (-50) for i, idx in enumerate(idxs)])
    Y = Y.T  # shape must be in shape (n_times, n_chans[, n_trials])
    c0, nc0 = tscov(Y)
    c1, nc1 = tscov(D)
    c0 /= nc0  # divide by total weight to normalize
    c1 /= nc1
    to_jd2, from_jd2, pwr0_biased, pwr1_biased = dss0(c0, c1)  # compute the transformations
    to_jd2 = to_jd2[:, np.argsort(pwr1_biased)[::-1]]  # sort them by magnitude
    from_jd2 = from_jd2[np.argsort(pwr1_biased)[::-1], :]
    return to_jd1, from_jd1, to_jd2, from_jd2


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
    to_jd1, from_jd1, to_jd2, from_jd2 = compute_transformation(epochs_from, conditions, keep, type=type)
    Y = apply_transform(epochs_to.get_data(), [to_jd1, to_jd2, from_jd2, from_jd1])
    transformed_epochs = epochs_to.copy()
    transformed_epochs._data = Y
    return transformed_epochs, to_jd1, from_jd1, to_jd2, from_jd2


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
keep = 10
evokeds_to, evokeds_jd, evokeds_avrgd_to, evokeds_avrgd_jd = \
    cfg["epochs"][f"event_id_{experiment}"].copy(), \
    cfg["epochs"][f"event_id_{experiment}"].copy(), \
    cfg["epochs"][f"event_id_{experiment}"].copy(), \
    cfg["epochs"][f"event_id_{experiment}"].copy(),
for key in cfg["epochs"][f"event_id_{experiment}"]:
    evokeds_to[key], evokeds_jd[key], evokeds_avrgd_to[key], evokeds_avrgd_to[key] = \
        list(), list(), list(), list()
channel_weights_all = []

for id in ids:
    epochs_from = get_epochs(id, "vocal_effort")
    epochs_to = get_epochs(id, "pinknoise")
    # epochs_to = epochs_from
    epochs_jd, to_jd1, from_jd1, to_jd2, from_jd2 = transform_epochs(epochs_from, epochs_to, condition_ids, keep, type="linear")

    evoked_jd = evoked_from_epochs(epochs_jd)
    evoked_from = evoked_from_epochs(epochs_from)
    evoked_to = evoked_from_epochs(epochs_to)
    evoked_diff = [mne.combine_evoked([evoked_jd[i-1], evoked_to[i-1]], weights=[1, -1]) for i in condition_ids]

    for condition in evoked_to:
        evokeds_to[condition.comment].append(condition)
        if len(evokeds_to[condition.comment]) == len(ids):
            evokeds_avrgd_to[condition.comment] = mne.grand_average(
                evokeds_to[condition.comment])
    for condition in evoked_jd:
        evokeds_jd[condition.comment].append(condition)
        if len(evokeds_jd[condition.comment]) == len(ids):
            evokeds_avrgd_jd[condition.comment] = mne.grand_average(
                evokeds_jd[condition.comment])
    channel_weights = from_jd2 @ from_jd1
    channel_weights_all.append(channel_weights)

# mne.viz.plot_compare_evokeds(evoked_from, picks="FCz", title="JD from")
# mne.viz.plot_compare_evokeds(evoked_to, picks="FCz", title="JD to")
# mne.viz.plot_compare_evokeds(evoked_jd, picks="FCz", title="Transformed")
# mne.viz.plot_compare_evokeds(evoked_diff, picks="FCz", title="Diff")

# evoked_diff = [mne.combine_evoked([evokeds_avrgd_to[key], evokeds_avrgd_jd[key]], weights=[1, -1]) for i in enumerate(evokeds_avrgd_to) if i != "deviant"]

mne.viz.plot_compare_evokeds(ignore_conds(evokeds_avrgd_to, "deviant", "button_press"), picks="FCz")
mne.viz.plot_compare_evokeds(ignore_conds(evokeds_avrgd_jd, "deviant", "button_press"), picks="FCz")
# mne.viz.plot_compare_evokeds(evoked_diff, picks="FCz")

channel_weights_avrgd = np.asarray(channel_weights_all).mean(axis=0)
mne.viz.plot_topomap(channel_weights_avrgd[0, :], epochs_jd.info)
mne.viz.plot_topomap(channel_weights_avrgd[1, :], epochs_jd.info)
mne.viz.plot_topomap(channel_weights_avrgd[2, :], epochs_jd.info)
