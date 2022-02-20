import slab
import pathlib
import os
from os import listdir
from os.path import isfile, join

DIR = pathlib.Path(os.getcwd())


def load_sounds(sound_type, room_dimensions):
    loaded_sound_object = {sound_type: {room_dimensions: {}}}
    sound_category = sound_type + '_' + 'room-' + room_dimensions
    a_weighted_filepath = DIR / 'experiment' / 'samples' / sound_category / 'a_weighted'
    file_names = [f for f in listdir(a_weighted_filepath) if
                  isfile(join(a_weighted_filepath, f)) and not f.startswith('.')]
    for file_name in file_names:
        file_path = a_weighted_filepath / file_name
        distance = file_name[file_name.find('dist-') + len('dist-'):file_name.rfind('.wav')]
        sound = slab.Binaural(file_path)
        loaded_sound_object[sound_type][room_dimensions][distance] = sound
    return loaded_sound_object
