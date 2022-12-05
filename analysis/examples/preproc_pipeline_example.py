import mne
import os
import pathlib
import numpy as np
import mne
from matplotlib import pyplot as plt
from autoreject import AutoReject, Ransac
from mne.preprocessing import ICA
import json
import glob
%matplotlib qt


# STEP 1: load the raw BrainVision data that comes out of the EEG setup and
# transfer it into a mne.Raw object. Do not stress about this first step too much,
# it has nothing to do with actual data processing but is required to get to the
# fundamental mne.Raw object. I simply put this first step into the example since
# this is what you first end up with as soon as you have gathered your data. In this
# example we actually preprocess the data of one of the pilot subjects that are uploaded
# in our google share drive. This is what makes this example really practical and
# should serve as a direct guide for your own data processing in the futue. Everything
# I mention in this example is also explained (more professionally) on the official
# MNE website (https://mne.tools/stable/auto_tutorials/preprocessing/index.html).
experiment = "noise"  # "laughter" or "noise" pilot data.
DIR = pathlib.Path(os.getcwd())
# get pilot folder directories.
pilot_DIR = DIR / "analysis" / "data" / f"pilot_{experiment}"
# get subject ids
ids = list(name for name in os.listdir(pilot_DIR)
           if os.path.isdir(os.path.join(pilot_DIR, name)))
id = ids[0]  # pick first subject for demonstration purpose.
folder_path = pilot_DIR / id
header_files = folder_path.glob("*.vhdr")
raw_files = []
for header_file in header_files:
    raw_files.append(mne.io.read_raw_brainvision(
        header_file, preload=True))  # read BrainVision files.
raw = mne.concatenate_raws(raw_files)  # make raw file
mapping = {"1": "Fp1", "2": "Fp2", "3": "F7", "4": "F3", "5": "Fz", "6": "F4",
           "7": "F8", "8": "FC5", "9": "FC1", "10": "FC2", "11": "FC6",
           "12": "T7", "13": "C3", "14": "Cz", "15": "C4", "16": "T8", "17": "TP9",
           "18": "CP5", "19": "CP1", "20": "CP2", "21": "CP6", "22": "TP10",
           "23": "P7", "24": "P3", "25": "Pz", "26": "P4", "27": "P8", "28": "PO9",
           "29": "O1", "30": "Oz", "31": "O2", "32": "PO10", "33": "AF7", "34": "AF3",
           "35": "AF4", "36": "AF8", "37": "F5", "38": "F1", "39": "F2", "40": "F6",
           "41": "FT9", "42": "FT7", "43": "FC3", "44": "FC4", "45": "FT8", "46": "FT10",
           "47": "C5", "48": "C1", "49": "C2", "50": "C6", "51": "TP7", "52": "CP3",
           "53": "CPz", "54": "CP4", "55": "TP8", "56": "P5", "57": "P1", "58": "P2",
           "59": "P6", "60": "PO7", "61": "PO3", "62": "POz", "63": "PO4", "64": "PO8"}
raw.rename_channels(mapping)  # Look at supplements below for mapping variable.
# Use BrainVision montage file to specify electrode positions.
montage_path = DIR / "analysis" / "AS-96_REF.bvef"
montage = mne.channels.read_custom_montage(fname=montage_path)
raw.set_montage(montage)


# STEP 2: Inspect the Raw file, look at if the general continuous data you gathered
# looks fine. The Raw object is continuous time series data, you can see all the
# data from start to end of your recording there. You have the time on your x-axis,
# and all the 64 channels on the y-axis with all the amplitudes they were measuring.
# The plots are interactive so you can go through the whole recording and have a
# look at what you recorded. The different stimuli are already annotated in the raw
# data.
raw.plot()


# STEP 3: Filter the data. The raw data we have so far is full of electrical activity
# that we do not need. We can apply a bandpass filter between 1 - 40 Hz to broadly
# cut off frequencies that we are not interested in. Plot the power density spectrum
# to look at what energy lies in different frequency bands that might disturb the brain
# signal.
raw.plot_psd()


# Clearly, the power-line-noise is definitely overlapping the brain activity.
# We want to filter the data in order to get rid of all that stuff. A bandpass filter
# between 1 - 40 is a classical filtering approach. Look at the psd again after you
# filtered to see the difference.
raw.filter(1, 40)


# STEP 4: epoch the data. mne.Epochs objects are slices of the raw data, we can
# decide which parts before and after stimulus onset we want to have for our epochs.
# We first take all the events we have in the raw file. We further need the descriptions
# for all the events we have, named event_id. We can also set a baseline, which is
# not recommended for now. Look at the epoched data by plotting it.
events = mne.events_from_annotations(raw)[0]  # get events
tmin = -0.5
tmax = 1.5
event_id = {"deviant": 1,
            "control": 2,
            "distance/20": 3,
            "distance/200": 4,
            "distance/1000": 5,
            "distance/2000": 6,
            "button_press": 7}
epochs = mne.Epochs(raw,
                    events=events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                    preload=True)
del raw

# STEP 5: rereference the data. Since the EEG setup we have at the freefield lab
# uses the FCz electrode as reference and we expect highly auditory correlated
# activity at the center electrodes, a reference at the mastoids or an average over
# all electrodes would be best in our case. Since the auditory cortices lie (more or less)
# in the superior temporal gyrus, we have electrical poles at the top and bottom
# of the head. A reference should have close to zero activity in general, or atleast only
# activity that is unrelated to auditory activity. Remember: voltage is always a difference
# measure. Here, we use an average reference. Before we can take the average voltage
# over all electrodes we have to make sure that every electrode we include is actually
# not recording nonsense activity. We use the RANSAC algorithm for that purpose.
epochs.plot_sensors(kind="topomap", ch_type='all')
reference = "average"
ransac = Ransac(n_jobs=-1)
epochs = ransac.fit_transform(epochs)
epochs.set_eeg_reference(ref_channels=reference)

# STEP 6: apply ICA to remove statistically brain-unrelated activity,
# such as eye saccades and blinks. We can plot the independent components to make
# sure that these are really independent.
ica = ICA()
ica.fit(epochs)
ica.plot_components(picks=range(10))
ica.plot_sources(epochs)
ica.apply(epochs, exclude=[1, 2])

# STEP 7: use the AutoReject algorithm to automatically detect and reject bad epochs
# trial-wise. The alrogithm automatically rejects epochs by computation of sensor-wise
# peak-to-peak-amplitude thresholds via cross-validation.
ar = AutoReject(n_jobs=-1)
epochs_ar, reject_log = ar.fit_transform(epochs)
# Visually inspect the data.
epochs_ar.plot_drop_log()
reject_log.plot_epochs(epochs)
fig, ax = plt.subplots(2)
epochs.average().plot(axes=ax[0])
epochs_ar.average().plot(axes=ax[1])

# STEP 8: get evoked responses out of epochs. Evoked responses are the averages of
# all epochs of one condition. These evoked responses (or Event-Related-Potentials)
# can then be used for data analysis and making fancy plots.
evokeds = [epochs_ar[condition].average() for condition in event_id.keys()]
evokeds[2].plot_joint(times="auto")
