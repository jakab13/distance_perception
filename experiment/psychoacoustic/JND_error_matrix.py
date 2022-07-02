import copy
import numpy
import slab
import os
import pathlib
import re
import random
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import time

############################################################################

subject_id = "jakab"
set_name = "log"
# set_name = "lin"
# set_name = "log_inv"
isi = 1.0
stage = "test"
# stage = "training_away"
# stage = "training_toward"

############################################################################

slab.set_default_samplerate(44100)
DIR = pathlib.Path(os.getcwd())

def create_and_store_file(parent_folder="confusion_matrices", subject_id="test", trialsequence=None, set_name=None):
    file = slab.ResultsFile(subject=subject_id, folder=parent_folder)
    file.write(subject_id, tag='subject_ID')
    file.write(set_name, tag='set_name')
    file.write(trialsequence, tag='trial')
    return file

def plot_matrix(subject_id, set_name):
    subject_folder = DIR / "confusion_matrices" / subject_id
    text_file_names = [f for f in listdir(subject_folder) if not f.endswith('.png') and not f.startswith('.')]
    text_file_names.sort()
    seq_data = slab.ResultsFile.read_file(subject_folder / text_file_names[-1], tag="trial")
    seq_data = seq_data["data"]
    y_test = [int(i[0]) for i in seq_data]
    y_pred = [int(i[1]) for i in seq_data]
    cm = confusion_matrix(y_test, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
    fig, ax = plt.subplots(figsize=(5, 5))
    sn.heatmap(cmn, annot=True, fmt='.2f', cmap="Blues")
    fig_title = "subject-{}_set-{}".format(subject_id, set_name)
    plt.title(fig_title + "_" + str(time.time()))
    plt.savefig(subject_folder / str(fig_title + ".png"))
    plt.show(block=False)


folder_path = DIR / 'experiment' / 'samples' / 'pinknoise_ramped' / 'normalised' / 'pydub'

file_names = [f for f in listdir(folder_path)
              if isfile(join(folder_path, f))
              and not f.startswith('.')]

loaded_sound_obj = {}

for file_name in file_names:
    file_path = folder_path / file_name
    distance_string = file_name[file_name.find('dist-') + len('dist-'):file_name.rfind('.wav')]
    distance = int(re.findall('\d+', distance_string)[0])
    loaded_sound_obj[distance] = slab.Binaural(file_path)

distance_sets = {
    'lin': {
        1: numpy.arange(20, 80, 20),
        2: numpy.arange(680, 760, 20),
        3: numpy.arange(1320, 1500, 20),
        4: numpy.arange(2000, 2200, 20),
        5: numpy.arange(2600, 3000, 20)
    },
    'log': {
        1: numpy.arange(20, 80, 20),
        2: numpy.arange(280, 340, 20),
        3: numpy.arange(780, 840, 20),
        4: numpy.arange(1320, 1400, 20),
        5: numpy.arange(2600, 3000, 20)
    },
    'log_inv': {
        1: numpy.arange(20, 80, 20),
        2: numpy.arange(900, 1000, 20),
        3: numpy.arange(1300, 1500, 20),
        4: numpy.arange(1800, 2000, 20),
        5: numpy.arange(2600, 3000, 20)
    }
}

response = 0
right_response = 0
distance_set = distance_sets[set_name]

if stage == "test":
    seq = slab.Trialsequence(conditions=[1, 2, 3, 4, 5], n_reps=2)
elif stage == "training_away":
    seq = slab.Trialsequence(conditions=[1, 2, 3, 4, 5], trials=[1, 2, 3, 4, 5], n_reps=1)
elif stage == "training_toward":
    seq = slab.Trialsequence(conditions=[1, 2, 3, 4, 5], trials=[5, 4, 3, 2, 1], n_reps=1)

for group in seq:
    distance = random.choice(distance_set[group])
    sound = loaded_sound_obj[distance]
    out = copy.deepcopy(sound)
    out.data = out.data[:slab.Signal.in_samples(isi, 44100)]
    out = out.ramp(duration=0.01)
    out.play()
    seq.add_response(group)
    if stage == "test":
        with slab.key('Button press') as key:
            response = key.getch() - 48
            seq.add_response(response)
        print(str(right_response) + ' / ' + str(seq.this_n))
        print('playing group', group)
        print('Response:', response)

if stage == "test":
    create_and_store_file(subject_id=subject_id, trialsequence=seq, set_name=set_name)
    plot_matrix(subject_id, set_name)