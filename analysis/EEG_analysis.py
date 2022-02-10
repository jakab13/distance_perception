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