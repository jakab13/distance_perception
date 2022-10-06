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

    to_jd1, from_jd1, _, pwr = dss1(X)  # compute the transformations
    del X
    to_jd1 = to_jd1[:, np.argsort(pwr)[::-1]]  # sort them by magnitude
    from_jd1 = from_jd1[np.argsort(pwr)[::-1], :]
    to_jd1 = to_jd1[:, 0:keep]  # only keep the largest ones
    from_jd1 = from_jd1[0:keep, :]

    Y = apply_transform(epochs.get_data(), to_jd1)  # apply the unmixing matrix to get the components

    idx1 = np.where(events[:, 2] == condition1)[0]
    idx2 = np.where(events[:, 2] == condition2)[0]
    D = Y[idx1, :, :].mean(axis=0) - Y[idx2, :, :].mean(axis=0)    # compute the difference between conditions
    Y, D = Y.T, D.T  # shape must be in shape (n_times, n_chans[, n_trials])
    c0, nc0 = tscov(Y)
    c1, nc1 = tscov(D)
    c0 /= nc0  # divide by total weight to normalize
    c1 /= nc1
    to_jd2, from_jd2, _, pwr = dss0(c0, c1)  # compute the transformations
    to_jd2 = to_jd2[:, np.argsort(pwr)[::-1]]  # sort them by magnitude
    from_jd2 = from_jd2[np.argsort(pwr)[::-1], :]

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
condition_2 = 5
keep = 8
# initialise evokeds and related objects
evokeds, evokeds_avrgd, evokeds_data = cfg["epochs"][f"event_id_{experiment}"].copy(
    ), cfg["epochs"][f"event_id_{experiment}"].copy(), cfg["epochs"][f"event_id_{experiment}"].copy()
for key in cfg["epochs"][f"event_id_{experiment}"]:
    evokeds[key], evokeds_avrgd[key], evokeds_data[key] = list(), list(), list()

evoked_jds = [[], [], [], [], []]
evoked_jds_avrgd = [[], [], [], [], []]

for id in ids:
    epochs_folder_VE = VE_DIR / id / "epochs"
    epochs_VE = mne.read_epochs(epochs_folder_VE / pathlib.Path(id + '-epo.fif'))
    epochs_folder_PN = PN_DIR / id / "epochs"
    epochs_PN = mne.read_epochs(epochs_folder_PN / pathlib.Path(id + '-epo.fif'))
    epochs_PN.apply_baseline(baseline=(-0.2, 0))
    epochs_PN.shift_time(-0.1, relative=True)

    # epochs_VE.crop(tmin=0.2, tmax=0.3)
    to_jd1, from_jd1, to_jd2, from_jd2 = compute_transformation(epochs_VE, condition_1, condition_2, keep)
    Y = apply_transform(epochs_PN.get_data(), [to_jd1, to_jd2])
    idx1 = np.where(epochs_PN.events[:, 2] == 1)[0]
    idx2 = np.where(epochs_PN.events[:, 2] == 2)[0]
    idx3 = np.where(epochs_PN.events[:, 2] == 3)[0]
    idx4 = np.where(epochs_PN.events[:, 2] == 4)[0]
    idx5 = np.where(epochs_PN.events[:, 2] == 5)[0]
    evoked_jd = [Y[idx1, 0, :].mean(axis=0),
                 Y[idx2, 0, :].mean(axis=0),
                 Y[idx3, 0, :].mean(axis=0),
                 Y[idx4, 0, :].mean(axis=0),
                 Y[idx5, 0, :].mean(axis=0)]
    for condition in [0, 1, 2, 3, 4]:
        evoked_jds[condition].append(np.asarray(evoked_jd[condition]))

for condition in [0, 1, 2, 3, 4]:
    evoked_jds_avrgd[condition] = np.asarray(evoked_jds[condition]).mean(axis=0)

def plot_evoked_jds(jds):
    x = np.arange(0, len(jds[0]))
    for idx, jd in enumerate(jds):
        name = f"{experiment}/" + str(idx + 1)
        plt.plot(x, -jd, label=name)
    plt.title("Pinknoise ERP - Beamformed from original Pinknoise results")
    plt.legend()
    plt.show()

plot_evoked_jds(evoked_jds_avrgd)