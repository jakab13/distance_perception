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

n_blocks = 2 # number of trial-blocks recorded per participant to stitch together
data_path = '/Users/paulfriedrich/Desktop/eeg/eeg_data/2022_02_04_pinknoise_jakab/'
file_name = '04022022_Pinknoise_Jakab_trial' # file name: ddmmyyyy_expname_subjname_blocknr
# eg '04022022_Pinknoise_Jakab_1.vhdr'

# concatenate raw files for n_blocks
raws = []
events_list = []
for block in range(n_blocks):
    raw = mne.io.read_raw_brainvision(data_path + file_name + str(block+1) + '.vhdr', preload=True)
    annotation = mne.read_annotations(data_path + file_name + str(block+1) + '.vmrk')
    events, _ = mne.events_from_annotations(raw)
    raws.append(raw)
    events_list.append(events)
raw, events = mne.concatenate_raws(raws, preload=True, events_list=events_list, on_mismatch='raise', verbose=None)
event_id = dict(dist_1=1, dist_2=2, dist_3=3, dist_4 = 4, dist_5=5, dist_6=6, dist_7=7)  # use names you find intuitive
sr = raw.info['sfreq']