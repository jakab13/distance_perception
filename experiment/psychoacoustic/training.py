import copy
import slab
import os
import pathlib
import re
import random
from datetime import datetime
from os import listdir
from os.path import isfile, join

#training:

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
    5: [3000]
}

isi = 1.2

training_away = slab.Trialsequence(conditions=[1, 2, 3, 4, 5], n_reps=1, trials=[1,2,3,4,5])
for group in training_away:
    distance = random.choice(groups[group])
    sound = loaded_sound_obj[distance]
    out = copy.deepcopy(sound)
    out.data = out.data[:slab.Sound.in_samples(isi, 44100)]
    out.level = 80
    out = out.ramp(duration=0.01)
    out.play()
    print("Finished")
print(training_away)

training_towards = slab.Trialsequence(conditions = [1,2,3,4,5], n_reps=1, trials=[5,4,3,2,1])
for group in training_towards:
    distance = random.choice(groups[group])
    sound = loaded_sound_obj[distance]
    out = copy.deepcopy(sound)
    out.data = out.data[:slab.Sound.in_samples(isi, 44100)]
    out.level = 80
    out = out.ramp(duration=0.01)
    out.play()
    print("Finished")
print(training_towards)