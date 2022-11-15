import pathlib
import os
import json
import mne
from analysis.utils import apply_transform
import numpy as np
from meegkit.dss import dss0, dss1
from meegkit.utils.covariances import tscov
import matplotlib.pyplot as plt


def compute_transformation(epochs, condition1, condition2, keep):

    if not (condition1 in epochs.events[:, 2] and condition2 in epochs.events[:, 2]):
        raise ValueError("'conditions' must be values of two event types!")
    X = epochs.get_data().transpose(2, 1, 0)
    events = epochs.events

    to_jd1, from_jd1, _, pwr = dss1(X, keep1=keep)  # compute the transformations
    del X

    Y = apply_transform(epochs.get_data(), to_jd1)  # apply the unmixing matrix to get the components

    idx1 = np.where(events[:, 2] == condition1)[0]
    idx2 = np.where(events[:, 2] == condition2)[0]
    D = Y[idx1, :, :].mean(axis=0) - Y[idx2, :, :].mean(axis=0)    # compute the difference between conditions
    Y, D = Y.T, D.T  # shape must be in shape (n_times, n_chans[, n_trials])
    c0, nc0 = tscov(Y)
    c1, nc1 = tscov(D)
    c0 /= nc0  # divide by total weight to normalize
    c1 /= nc1
    to_jd2, from_jd2, _, pwr = dss0(c0, c1, keep1=keep)  # compute the transformations

    return to_jd1, from_jd1, to_jd2, from_jd2

experiment = "pinknoise"  # either "pinknoise" or "vocal_effort"
# get pilot folder directory.
DIR = pathlib.Path(os.getcwd())
VE_DIR = DIR / "analysis" / "data" / "vocal_effort"  # pilot_laughter or pilot_noise
PN_DIR = DIR / "analysis" / "data" / "pinknoise"
with open(DIR / "analysis" / "preproc_config.json") as file:
    cfg = json.load(file)
# get subject ids
ids_VE = list(name for name in os.listdir(VE_DIR) if os.path.isdir(os.path.join(VE_DIR, name)))
ids_PN = list(name for name in os.listdir(PN_DIR) if os.path.isdir(os.path.join(PN_DIR, name)))
ids = list(set(ids_VE) & set(ids_PN))
condition_1 = 1
condition_2 = 2
condition_3 = 3
condition_4 = 4
condition_5 = 5
keep = 12
# initialise evokeds and related objects
evokeds, evokeds_avrgd, evokeds_data = cfg["epochs"][f"event_id_{experiment}"].copy(
    ), cfg["epochs"][f"event_id_{experiment}"].copy(), cfg["epochs"][f"event_id_{experiment}"].copy()
for key in cfg["epochs"][f"event_id_{experiment}"]:
    evokeds[key], evokeds_avrgd[key], evokeds_data[key] = list(), list(), list()

for id in ids:
    epochs_folder_VE = VE_DIR / id / "epochs"
    epochs_VE = mne.read_epochs(epochs_folder_VE / pathlib.Path(id + '-epo.fif'))
    epochs_folder_PN = PN_DIR / id / "epochs"
    epochs_PN = mne.read_epochs(epochs_folder_PN / pathlib.Path(id + '-epo.fif'))
    epochs_PN.apply_baseline(baseline=(-0.2, 0))
    epochs_PN.shift_time(-0.1, relative=True)

    # epochs_VE.crop(tmin=0.2, tmax=0.3)
    to_jd1, from_jd1, to_jd2, from_jd2 = compute_transformation(epochs_VE, condition_5, condition_1, keep)
    Y = apply_transform(epochs_PN.get_data(), [to_jd1, to_jd2, from_jd2, from_jd1])
    epochs_PN_jd = epochs_PN.copy()
    epochs_PN_jd._data = Y
    evoked = [epochs_PN_jd[condition].average()
              for condition in cfg["epochs"][f"event_id_{experiment}"].keys()]
    for condition in evoked:
        if condition.comment in evokeds:
            evokeds[condition.comment].append(condition)
            if len(evokeds[condition.comment]) == len(ids):
                evokeds_avrgd[condition.comment] = mne.grand_average(
                    evokeds[condition.comment])
            else:
                continue


