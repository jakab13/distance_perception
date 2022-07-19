import copy
import time

import numpy
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

#################################################################

subject_id = 'joschua'
current_duration = 150
vocalist = 'vocalist-2'
is_training = True

#################################################################

if is_training:
    current_duration = 300

def create_and_store_file(parent_folder, subject_id, trialsequence, duration, vocalist):
    file = slab.ResultsFile(subject=subject_id, folder=parent_folder)
    subject_id = subject_id
    file.write(subject_id, tag='subject_ID')
    today = datetime.now()
    file.write(today.strftime('%Y/%m/%d'), tag='Date')
    file.write(today.strftime('%H:%M:%S'), tag='Time')
    file.write(duration, tag='duration')
    file.write(trialsequence, tag='trial')
    file.write(vocalist, tag='vocalist')
    return file

slab.set_default_samplerate(44100)
DIR = pathlib.Path(os.getcwd())

folder_path = DIR / 'experiment' / 'samples' / 'vocal_effort' / vocalist / 'pyloudnorm' / str(current_duration)

file_names = [f for f in listdir(folder_path)
              if isfile(join(folder_path, f))
              and not f.startswith('.')]

loaded_sound_obj = {1: None, 2: None, 3: None, 4: None, 5: None}

for file_name in file_names:
    file_path = folder_path / file_name
    distance_string = file_name[file_name.find('dist-') + len('dist-'):file_name.rfind('_try')]
    distance = int(re.findall('\d+', distance_string)[0])
    sound = slab.Binaural(file_path)
    if loaded_sound_obj[distance] is None:
        loaded_sound_obj[distance] = [sound]
    else:
        loaded_sound_obj[distance].append(sound)

isi = 0.7
n = 1
response = 0
right_response = 0

if is_training:
    seq = slab.Trialsequence(trials=[1, 2, 3, 4, 5, 5, 4, 3, 2, 1], n_reps=10)
else:
    seq = slab.Trialsequence(conditions=[1, 2, 3, 4, 5], n_reps=10)
for group in seq:
    sound = random.choice(loaded_sound_obj[group])
    out = copy.deepcopy(sound)
    out.level = 80
    out = out.ramp(duration=0.01)
    out.play()
    if not is_training:
        seq.add_response(group)
        with slab.key('Button press') as key:
            response = key.getch() - 48
            seq.add_response(response)
        if response == group:
            seq.add_response(1)
            right_response += 1
        else:
            seq.add_response(0)
            # print out live result
        print(str(right_response) + ' / ' + str(n))
        n += 1
        responses = seq.save_json("sequence.json", clobber=True)
        print("Finished")
        print('playing group', group)
        print('Response:', response)
    time.sleep(0.8) #check out best timing

if not is_training:
    create_and_store_file(parent_folder=DIR / 'experiment' / 'results', subject_id=subject_id,
                                          trialsequence=seq, duration=current_duration, vocalist=vocalist)
