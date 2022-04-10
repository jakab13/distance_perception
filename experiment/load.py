import slab
import pathlib
import os
from os import listdir
from os.path import isfile, join
import random
import string

DIR = pathlib.Path(__file__).parent.absolute()


def generate_id():
    characters = string.ascii_lowercase + string.digits
    participant_id = ''.join(random.choice(characters) for i in range(6))
    print("Participant ID is:", participant_id)
    return participant_id


def load_control(sound_type):
    a_weighted_filepath = DIR / 'samples' / sound_type / 'a_weighted'
    control_file_name = 'AW_A_' + sound_type + '_control.wav'
    control_file_path = a_weighted_filepath / control_file_name
    control_sound = slab.Binaural(control_file_path)
    return control_sound


def load_deviant():
    deviant_filepath = DIR / 'samples' / 'chirp' / 'a_weighted' / 'AW_A_chirp_control.wav'
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
    for file_name in file_names:
        file_path = a_weighted_filepath / file_name
        sound = slab.Binaural(file_path)
        distance = file_name[file_name.find('dist-') + len('dist-'):file_name.rfind('.wav')]
        loaded_sound_obj[sound_type][distance] = sound

    loaded_sound_obj[sound_type]['control'] = load_control(sound_type)
    loaded_sound_obj['deviant'] = load_deviant()
    return loaded_sound_obj
