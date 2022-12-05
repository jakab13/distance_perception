import copy
import slab
import os
import pathlib
import re
import random
from os import listdir
from os.path import isfile, join

slab.set_default_samplerate(44100)
DIR = pathlib.Path(os.getcwd())

folder_path = DIR / 'experiment' / 'samples' / 'USOs' / 'USOs'

file_names = [f for f in listdir(folder_path)
              if isfile(join(folder_path, f))
              and not f.startswith('.')]

loaded_sound_obj = {}

for file_name in file_names:
    file_path = folder_path / file_name
    sound_id_string = file_name[file_name.find('300ms_') + len('300ms_'):file_name.rfind('_room')]
    sound_id = int(re.findall('\d+', sound_id_string)[0])
    distance_string = file_name[file_name.find('dist-') + len('dist-'):file_name.rfind('.wav')]
    distance = int(re.findall('\d+', distance_string)[0])
    if distance not in loaded_sound_obj:
        loaded_sound_obj[distance] = {}
    if sound_id not in loaded_sound_obj[distance]:
        loaded_sound_obj[distance][sound_id] = slab.Binaural(file_path)
    else:
        loaded_sound_obj[distance][sound_id].append(slab.Binaural(file_path))

groups = {
    1: [17, 18, 19, 20, 21, 22, 23],
    2: [57, 60, 64, 67, 70, 74, 77],
    3: [190, 201, 212, 224, 235, 246, 257],
    4: [636, 673, 710, 748, 785, 822, 860],
    5: [2125, 2250, 2375, 2500, 2625, 2750, 2875]
}


sound_id = 4
isi = 0.7

seq = slab.Trialsequence(conditions=[1, 2, 3, 4, 5], trials=[1, 2, 3, 4, 5])
for group in seq:
    distance = random.choice(groups[group])
    sound = loaded_sound_obj[distance][sound_id]
    out = copy.deepcopy(sound)
    out.data = out.data[:slab.Sound.in_samples(isi, 44100)]
    out.level = 80
    out = out.ramp(duration=0.01)
    out.play()
