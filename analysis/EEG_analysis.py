import os
import pathlib
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

DIR = pathlib.Path(os.getcwd())

folder_name = '2022_02_04_pinknoise_jakab'
folder_path = DIR / 'data' / folder_name

header_files = folder_path.glob('*.vhdr')
raw_files = []

for idx, header_file in enumerate(header_files):
    raw_files.append(mne.io.read_raw_brainvision(header_file))

raw = mne.concatenate_raws(raw_files, verbose=True)

# get hit rate for button presses
# for every sweep onset, check if there was a button pressed within
# a time window of ISI
ISI = sr * 2
sweep_times = events[np.where(events[:, 2] == 1)][:, 0]
button_times = events[np.where(events[:, 2] == 7)][:, 0]
hits = 0
misses = 0
for sweep_time in sweep_times:
    interval = sweep_time + 3 * sr # 3 sec interval after button press
    if np.where(np.logical_and(button_times >= sweep_time, button_times <= interval)) is not None:
        hits = +1
    else:
        misses = +1