import copy
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

def create_and_store_file(parent_folder, subject_folder, subject_id, trialsequence, group):
    file = slab.ResultsFile(subject=subject_folder, folder=parent_folder)
    subject_id = subject_id
    file.write(subject_id, tag='subject_ID')
    today = datetime.now()
    file.write(today.strftime('%Y/%m/%d'), tag='Date')
    file.write(today.strftime('%H:%M:%S'), tag='Time')
    file.write(group, tag='group')
    file.write(trialsequence, tag='Trial')
    return file

slab.set_default_samplerate(44100)
DIR = pathlib.Path(os.getcwd())

folder_path = DIR / 'experiment' / 'samples' / 'pinknoise_ramped' / 'normalised' / 'pyloudnorm'

file_names = [f for f in listdir(folder_path)
              if isfile(join(folder_path, f))
              and not f.startswith('.')]

loaded_sound_obj = {}

for file_name in file_names:
    file_path = folder_path / file_name
    distance_string = file_name[file_name.find('dist-') + len('dist-'):file_name.rfind('.wav')]
    distance = int(re.findall('\d+', distance_string)[0])
    loaded_sound_obj[distance] = slab.Binaural(file_path)

groups = {
    1: [20],
    2: [60],
    3: [220],
    4: [760],
    5: [2500]
}

isi = 0.7
n = 1
response = 0
right_response = 0

seq = slab.Trialsequence(conditions=[1, 2, 3, 4, 5], n_reps=10)
for group in seq:
    distance = random.choice(groups[group])
    sound = loaded_sound_obj[distance]
    out = copy.deepcopy(sound)
    out.data = out.data[:slab.Sound.in_samples(isi, 44100)]
    out.level = 80
    out = out.ramp(duration=0.01)
    out.play()
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
print(seq)

create_and_store_file(parent_folder='first_tries', subject_folder='Joschua', subject_id='jg',
                                          trialsequence=seq, group=groups)

seq_data = slab.ResultsFile.read_file(DIR / 'first_tries' / 'Joschua' / 'Joschua_2022-07-01-12-32-03.txt', tag="Trial")
seq_data = seq_data["data"]
y_test = [int(i[0]) for i in seq_data]
y_pred = [int(i[1]) for i in seq_data]


def plot_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    fig, ax = plt.subplots(figsize=(5, 5))
    sn.heatmap(cmn, annot=True, fmt='.2f', cmap="Blues")
    plt.show(block=False)


plot_matrix(y_test, y_pred)

