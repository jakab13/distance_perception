import slab
import pathlib
import os
from os import listdir
from os.path import isfile, join
import random
import string

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DIR = pathlib.Path(__file__).parent.absolute()


def generate_id():
    characters = string.ascii_lowercase + string.digits
    participant_id = ''.join(random.choice(characters) for i in range(6))
    print("Participant ID is:", participant_id)
    return participant_id


def load_controls(sound_type):
    a_weighted_filepath = DIR / 'samples' / sound_type / 'a_weighted'
    file_names = [f for f in listdir(a_weighted_filepath)
                  if isfile(join(a_weighted_filepath, f))
                  and not f.startswith('.')
                  and f.endswith('control.wav')]
    control_sounds = []
    for file_name in file_names:
        control_file_path = a_weighted_filepath / file_name
        control_sound = slab.Binaural(control_file_path)
        control_sounds.append(control_sound)
    return control_sounds


def load_deviant():
    deviant_filepath = DIR / 'samples' / 'chirp_room-10-30-3' / 'a_weighted' / 'AW_A_chirp_room-10-30-3_control.wav'
    deviant_sound = slab.Binaural(deviant_filepath)
    return deviant_sound


def load_sounds(sound_type):
    loaded_sound_obj = {
        sound_type: {
        }
    }
    a_weighted_filepath = DIR / 'samples' / sound_type / 'a_weighted'
    file_names = [f for f in listdir(a_weighted_filepath)
                  if isfile(join(a_weighted_filepath, f))
                  and not f.startswith('.')
                  and not f.endswith('control.wav')]
    file_names.sort()
    for file_name in file_names:
        file_path = a_weighted_filepath / file_name
        sound = slab.Binaural(file_path)
        distance = int(file_name[file_name.find('dist-') + len('dist-'):file_name.rfind('.wav')])
        if distance not in loaded_sound_obj[sound_type]:
            loaded_sound_obj[sound_type][distance] = [sound]
        else:
            loaded_sound_obj[sound_type][distance].append(sound)

    loaded_sound_obj[sound_type]['controls'] = load_controls(sound_type)
    loaded_sound_obj[sound_type][0] = [load_deviant()]
    return loaded_sound_obj
