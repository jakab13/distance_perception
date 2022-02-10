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
    raw_files.append(mne.io.read_raw_brainvision(header_file))

raw = mne.concatenate_raws(raw_files, verbose=True)
events = mne.events_from_annotations(raw)

fs = raw.info['sfreq']
# get hit rate for button presses
# for every sweep onset, check if there was a button pressed within
# a time window of ISI
ISI = fs * 2
event_arr = events[0]
event_idx = event_arr[:, 2]
event_times = event_arr[:, 0]
sweep_times = event_times[np.where(event_idx == 1)]
button_times = event_times[np.where(event_idx == 7)]
hits = []
misses = []
for sweep_time in sweep_times:
    interval = sweep_time + ISI  # ISI (2 sec interval after button press)
    button_in_interval = np.where(np.logical_and(button_times >= sweep_time, button_times <= interval))
    if button_in_interval is not None:
        hits.append(events[0])
    else:
        misses = +1