import pathlib
import os
from os.path import join
import mne
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

DIR = pathlib.Path(os.getcwd())

evokeds_folder = DIR / 'analysis' / 'data' / 'USOs' / 'evokeds'

def get_file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if not f.startswith('.'):
                yield pathlib.Path(join(dirpath, f))


evokeds_paths = [f for f in get_file_paths(evokeds_folder)]
evokeds = [mne.combine_evoked(mne.read_evokeds(f), weights='equal') for f in evokeds_paths]
for idx, evoked in enumerate(evokeds):
    evoked.comment = 'distance/' + str(idx+1)
    evoked.shift_time(-0.2, relative=True)
    evoked.data = np.roll(evoked.data, 100, axis=1)
    evoked.crop(tmin=-0.2, tmax=0.5)
combined_evokeds = mne.combine_evoked(evokeds, weights='equal')
combined_evokeds.plot_joint(times=[0.096, 0.206, 0.335])
mne.viz.plot_compare_evokeds(evokeds, colors=["#8ecae6", "#219ebc", "#023047", "#ffb703", "#fb8500"],
                             legend="upper right")
