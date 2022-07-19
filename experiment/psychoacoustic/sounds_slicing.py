import copy
import numpy as np
import slab
import os
import pathlib
import re
import random
from datetime import datetime
from os import listdir
from os.path import isfile, join
# from Function_Setup import create_and_store_file
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


slab.set_default_samplerate(44100)
DIR = pathlib.Path(os.getcwd()) #base of folders is current working directory


folder_path = DIR / 'experiment' / 'samples' / 'vocal_effort' / 'vocalist-11' / 'pyloudnorm' / '300' #change to vocalist 2 / 11 here


file_names = [pathlib.Path(folder_path / f) for f in listdir(folder_path)
              if isfile(join(folder_path, f))
              and not f.startswith('.')]

lengths = [0.25, 0.2, 0.15, 0.1, 0.05]

#steps: load the sound files, shorten them by 50ms, save, short again, save etc.
for file in file_names:
    sound = slab.Binaural(file)
    samplerate = sound.samplerate
    for length in lengths:
        length_in_samples = slab.Signal.in_samples(length, samplerate)
        sound.data = sound.data[:length_in_samples]
        length_string = str(int(length * 1000))
        folder_path_length = folder_path.parent / length_string
        if not os.path.exists(folder_path_length):
            os.makedirs(folder_path_length)
        sound.write(folder_path_length / str(file.stem + '_length-' + length_string + '.wav'))




