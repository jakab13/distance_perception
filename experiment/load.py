import slab
import pathlib
import time
import os
import re
from os import listdir
from os.path import isfile, join


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DIR = pathlib.Path(__file__).parent.absolute()


def load_controls(sound_type):
    a_weighted_filepath = DIR / 'samples' / sound_type
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
    deviant_sound = slab.Precomputed([slab.Binaural(deviant_filepath)])
    return deviant_sound


def load_sounds(sound_type):
    sound_type = sound_type
    loaded_sound_obj = {
        sound_type: {
        }
    }
    a_weighted_filepath = DIR / 'samples' / sound_type
    file_names = [f for f in listdir(a_weighted_filepath)
                  if isfile(join(a_weighted_filepath, f))
                  and not f.startswith('.')
                  and not f.endswith('control.wav')]
    file_names.sort()
    for file_name in file_names:
        file_path = a_weighted_filepath / file_name
        if sound_type == 'USOs_resampled':
            sound_id_string = file_name[file_name.find('300ms_') + len('300ms_'):file_name.rfind('_room')]
            sound_id = int(re.findall('\d+', sound_id_string)[0])
        else:
            sound_id = 0
        distance_string = file_name[file_name.find('dist-') + len('dist-'):file_name.rfind('.wav')]
        distance = int(re.findall('\d+', distance_string)[0])
        if distance not in loaded_sound_obj[sound_type]:
            loaded_sound_obj[sound_type][distance] = {}
        if sound_id not in loaded_sound_obj[sound_type][distance]:
            loaded_sound_obj[sound_type][distance][sound_id] = slab.Binaural(file_path)

    loaded_sound_obj[sound_type]['controls'] = load_controls(sound_type)
    loaded_sound_obj[sound_type][0] = load_deviant()
    return loaded_sound_obj
