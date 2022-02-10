import os
import pathlib
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

DIR = pathlib.Path(os.getcwd())
folder_name = '2022_02_04_pinknoise_jakab'
folder_path = DIR / 'analysis' / 'data' / folder_name

header_files = folder_path.glob('*.vhdr')
raw_files = []

for idx, header_file in enumerate(header_files):
    raw_files.append(mne.io.read_raw_brainvision(header_file, preload=True))

raw = mne.concatenate_raws(raw_files)
events = mne.events_from_annotations(raw)

raw.filter(0.5, 40)

ica = mne.preprocessing.ICA(n_components=0.99, method="fastica")
ica.fit(raw)

tmin = -0.3
tmax = 0.7
baseline = (-0.2, 0)
drops=[]
reject_criteria = dict(eeg=200e-6)
flat_criteria = dict(eeg=1e-6)
event_id = dict(deviant=1, control=2, dist_2m=3, dist_4m=4, dist_8m=5, dist_16m=6, button_press=7)
epochs = mne.Epochs(raw, events[0], event_id, tmin, tmax, reject=reject_criteria, flat=flat_criteria,
                    reject_by_annotation=True, baseline=baseline, preload=True)

deviant = epochs['deviant'].average()
control = epochs['control'].average()
dist_2m = epochs['dist_2m'].average()
dist_4m = epochs['dist_4m'].average()
dist_8m = epochs['dist_8m'].average()
dist_16m = epochs['dist_16m'].average()

mne.viz.plot_compare_evokeds([
                              dist_2m,
                              dist_4m,
                              dist_8m,
                              dist_16m
                              ])
