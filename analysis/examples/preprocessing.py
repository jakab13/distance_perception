from pathlib import Path
import argparse
import json
import numpy as np
from mne.io import read_raw_fif
from mne.epochs import Epochs
from autoreject import Ransac, AutoReject
from mne.preprocessing import ICA, read_ica, corrmap
from meegkit.dss import dss_line

root = Path(__file__).parent.parent.absolute()

parser = argparse.ArgumentParser(description="Preprocess a block of raw data. Saves 3 epoch files : \n one for the "
                                             "pre-target, one for the post target and one for all other epochs.")
parser.add_argument('experiment', type=str, choices=["freefield", "headphones"])
parser.add_argument('subject', type=str)
parser.add_argument('block', type=int)
parser.add_argument('config', type=str, help="Path to the config file containing the preprocessing parameters.")
parser.add_argument('--n_jobs', type=int, default=1, help="Number of multiprocessing jobs")
args = parser.parse_args()

try:
    cfg = json.load(open(args.config))
except FileNotFoundError:
    raise FileNotFoundError("'config' must be the path to a .JSON file!")

in_folder = root/args.experiment/"input"/"raw"/args.subject
assert in_folder.exists()
out_folder = root/args.experiment/"input"/"epochs"/args.subject
assert in_folder.exists()

# read raw data, events and stimulus sequence:
raw = read_raw_fif(in_folder/f"block{args.block}-raw.fif", preload=True)
events = np.loadtxt(in_folder/f"block{args.block}-eve.txt", dtype=int)
stimuli = np.loadtxt(in_folder/f"block{args.block}-seq.txt", dtype=int)

# STEP1: remove power line noise and highpass filter the data
X = raw.get_data().T
# remove power line noise with the zapline algorithm
X, _ = dss_line(X, fline=cfg["zapline"]["fline"], sfreq=raw.info["sfreq"], nremove=cfg["zapline"]["nremove"])
raw._data = X.T  # put the data back into raw
del X
raw = raw.filter(l_freq=cfg["filter"]["lfreq"], h_freq=None, phase=cfg["filter"]["phase"])

# STEP2: epoch and apply a robust reference to the raw data
epochs = Epochs(raw, events, tmin=cfg["epochs"]["tmin"], tmax=cfg["epochs"]["tmax"], baseline=None, preload=True)
r = Ransac(n_jobs=args.n_jobs)
epochs = r.fit_transform(epochs)
epochs.set_eeg_reference(ref_channels=cfg["reference"], projection=True)
average_reference = epochs.info["projs"]
epochs = Epochs(raw, events, event_id=cfg["epochs"]["event_ids"],
                tmin=cfg["epochs"]["tmin"], tmax=cfg["epochs"]["tmax"], baseline=None, preload=True)
epochs.add_proj(average_reference)
epochs.apply_proj()

# STEP 3: reject blink artifacts with ICA
reference = read_ica(str(root/cfg["ica"]["reference"]))
component = reference.labels_["blinks"]
ica = ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"])
ica.fit(epochs)
ica.labels_["blinks"] = []
corrmap([reference, ica], template=(0, component[0]), label="blinks", plot=False, threshold=cfg["ica"]["threshold"])
ica.apply(epochs, exclude=ica.labels_["blinks"])

# STEP4: reject or repair bad epochs (separately for pre-targets, post targets and others
idx = np.where(stimuli == 0)[0]
post_target_trials = idx - np.arange(0, len(idx))
target_trials = (idx - np.arange(0, len(idx))) - 1
other_trials = np.delete(np.array(range(len(events))),
                         np.concatenate([target_trials, post_target_trials]))
assert len(np.concatenate([target_trials, post_target_trials, other_trials])) == len(events)

for trials, name in zip([target_trials, post_target_trials, other_trials], ["targets", "post_targets", "block"]):
    if len(trials) < cfg["ar"]["cv"]:
        cv = len(trials)
    else:
        cv = cfg["ar"]["cv"]
    ar = AutoReject(cv=cv, n_interpolate=np.array(cfg["ar"]["n_interpolate"]), n_jobs=args.n_jobs)
    clean_epochs = ar.fit_transform(epochs[trials])

    clean_epochs.save(out_folder/f"{name}{args.block}-epo.fif", overwrite=True)

