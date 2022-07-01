import copy
import slab
import os
import pathlib
import re
import random
from datetime import datetime
from os import listdir
from os.path import isfile, join
from Function_Setup import create_and_store_file


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

seq = slab.Trialsequence(conditions=[1, 2, 3, 4, 5], n_reps=1)
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

