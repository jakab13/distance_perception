import slab
import pathlib
import os
from os import listdir
from os.path import isfile, join

DIR = pathlib.Path(os.getcwd())


def load_sounds(sound_type, room_dimensions):
    loaded_sound_object = {
        sound_type: {
            room_dimensions: {}
        }
    }
    sound_category = sound_type + '_' + 'room-' + room_dimensions
    a_weighted_filepath = DIR / 'experiment' / 'samples' / sound_category / 'a_weighted'
    file_names = [f for f in listdir(a_weighted_filepath) if
                  isfile(join(a_weighted_filepath, f)) and not f.startswith('.')]
    for file_name in file_names:
        file_path = a_weighted_filepath / file_name
        sound = slab.Binaural(file_path)
        distance = file_name[file_name.find('dist-') + len('dist-'):file_name.rfind('.wav')]
        loaded_sound_object[sound_type][room_dimensions][distance] = sound

    loaded_sound_object[sound_type][room_dimensions]['control'] = load_control(sound_type, room_dimensions)
    return loaded_sound_object


def load_control(sound_type, room_dimensions):
    sound_category = sound_type + '_' + 'room-' + room_dimensions
    a_weighted_filepath = DIR / 'experiment' / 'samples' / sound_category / 'a_weighted'
    control_file_name = 'AW_A_' + sound_category + '_control.wav'
    control_file_path = a_weighted_filepath / control_file_name
    control_sound = slab.Binaural(control_file_path)
    return control_sound


def load_deviant():
    deviant_filepath = DIR / 'experiment' / 'samples' / 'chirp_room-10-30-3' / 'a_weighted' / 'AW_A_chirp_room-10-30-3_control.wav'
    deviant_sound = slab.Binaural(deviant_filepath)
    return deviant_sound

