import os
import pathlib
import numpy as np
import mne
import matplotlib.pyplot as plt
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

DIR = pathlib.Path(os.getcwd())

sample_data_folder = '2022_02_04_pinknoise_jakab'
sample_data_raw_file = os.path.join(DIR / sample_data_folder /'2022_02_04_pinknoise_jakab_1.vhdr')

raw = mne.io.read_raw_brainvision(sample_data_raw_file)
