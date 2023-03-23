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
import numpy

#################################################################

subject_id = 'test'
ISI = 1.5

#################################################################

slab.set_default_samplerate(44100)
DIR = pathlib.Path(os.getcwd())

IACC_folder_path = DIR / 'experiment' / 'samples' / 'IACC_auralizations' / 'frontal'
castanets_folder_path = DIR / 'experiment' / 'samples' / 'IACC_auralizations' / 'room-large_castanets'
bark_folder_path = DIR / 'experiment' / 'samples' / 'bark_room-10-30-3' / 'normalised'
dunk_folder_path = DIR / 'experiment' / 'samples' / 'dunk_room-10-30-3' / 'normalised'
bum_folder_path = DIR / 'experiment' / 'samples' / 'bum_room-10-30-3' / 'normalised'


distance_sets = {
    'log_10': {
            1: numpy.arange(20, 60, 20),
            2: numpy.arange(140, 180, 20),
            3: numpy.arange(380, 440, 20),
            4: numpy.arange(560, 640, 20),
            5: numpy.arange(780, 860, 20),
            6: numpy.arange(980, 1020, 20),
            7: numpy.arange(1180, 1280, 20),
            8: numpy.arange(1480, 1560, 20),
            9: numpy.arange(1800, 2000, 20),
            10: numpy.arange(2300, 2500, 20)
        },
    'log_5': {
        1: numpy.arange(20, 60, 20),
        2: numpy.arange(380, 440, 20),
        3: numpy.arange(780, 860, 20),
        4: numpy.arange(1580, 1660, 20),
        5: numpy.arange(2300, 2500, 20)
    },
    'IACC_5': {
        1: numpy.arange(0, 0.05, 0.01),
        2: numpy.arange(0.15, 0.2, 0.01),
        3: numpy.arange(0.25, 0.3, 0.01),
        4: numpy.arange(0.35, 0.4, 0.01),
        5: numpy.arange(0.45, 0.5, 0.01)
    }
}

IACC_file_names = [f for f in listdir(IACC_folder_path)
              if isfile(join(IACC_folder_path, f))
              and not f.startswith('.')]
castanets_file_names = [f for f in listdir(castanets_folder_path)
              if isfile(join(castanets_folder_path, f))
              and not f.startswith('.')]
bark_file_names = [f for f in listdir(bark_folder_path)
              if isfile(join(bark_folder_path, f))
              and not f.startswith('.')]
dunk_file_names = [f for f in listdir(dunk_folder_path)
              if isfile(join(dunk_folder_path, f))
              and not f.startswith('.')]
bum_file_names = [f for f in listdir(bum_folder_path)
              if isfile(join(bum_folder_path, f))
              and not f.startswith('.')]


def play_distances(folder_path, direction="away", scale="log_10"):
    file_names = [f for f in listdir(folder_path)
                   if isfile(join(folder_path, f))
                   and not f.startswith('.')]
    conditions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if direction == "toward":
        conditions.reverse()
    for distance_group in conditions:
        distance = random.choice(distance_sets[scale][distance_group])
        distance_string = str(distance) + '.wav'
        file_name = [f for f in file_names if distance_string in f][0]
        sound = slab.Sound(folder_path / file_name)
        sound.data = sound.data[:int(sound.samplerate * ISI)]
        sound = sound.ramp()
        print("Playing from", str(distance/100) + 'm')
        sound.play()


#################################################################
# 1. Steven scale with IACC stimuli

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

#################################################################
#################################################################
# 2. Training with "realistic" distance stimuli

play_distances(bark_folder_path, direction="away")
play_distances(bark_folder_path, direction="toward")

play_distances(bum_folder_path, direction="away")
play_distances(bum_folder_path, direction="toward")

play_distances(dunk_folder_path, direction="away")
play_distances(dunk_folder_path, direction="toward")

#################################################################
# 3. IACC as distance

file = slab.ResultsFile(subject=subject_id)
IACC_seq2 = slab.Trialsequence(conditions=[1, 2, 3, 4, 5], n_reps=10)
for distance_group in IACC_seq2:
    IACC_val = round(random.choice(distance_sets['IACC_5'][distance_group]), 2)
    distance_string = str(IACC_val) + '_az'
    file_name = [f for f in castanets_file_names if distance_string in f][0]
    sound = slab.Sound(castanets_folder_path / file_name)
    sound.data = sound.data[:int(sound.samplerate * ISI)]
    sound = sound.ramp()
    sound.play()
    response = input()
    IACC_seq2.add_response(response)
    print("Playing IACC:", IACC_seq2.this_n + 1, "/", IACC_seq2.n_trials)

file.write(IACC_seq2, tag='IACC_distance')