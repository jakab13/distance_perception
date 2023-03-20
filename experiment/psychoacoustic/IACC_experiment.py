import copy
import time
import slab
import os
import pathlib
import re
import random
from datetime import datetime
from os import listdir
from os.path import isfile, join

#################################################################

subject_id = 'test'

#################################################################

slab.set_default_samplerate(44100)
DIR = pathlib.Path(os.getcwd())

IACC_folder_path = DIR / 'experiment' / 'samples' / 'IACC_auralizations' / 'frontal'
bark_folder_path = DIR / 'experiment' / 'samples' / 'bark_room-10-30-3' / 'normalised'
dunk_folder_path = DIR / 'experiment' / 'samples' / 'dunk_room-10-30-3' / 'normalised'
bum_folder_path = DIR / 'experiment' / 'samples' / 'bum_room-10-30-3' / 'normalised'

IACC_file_names = [f for f in listdir(IACC_folder_path)
              if isfile(join(IACC_folder_path, f))
              and not f.startswith('.')]

file_name_min = 'b_frontal_mix_0.wav'
file_name_max = 'b_frontal_mix_5.wav'

stim_min = slab.Sound(IACC_folder_path / file_name_min)
stim_min.play()

stim_max = slab.Sound(IACC_folder_path / file_name_max)
stim_max.play()

IACC_seq = slab.Trialsequence(IACC_file_names)

file = slab.ResultsFile(subject=subject_id)
for file_name in IACC_seq:
    sound = slab.Sound(IACC_folder_path / file_name)
    sound.play()
    response = input()
    IACC_seq.add_response(response)

file.write(IACC_seq, tag='stevens_scale')
