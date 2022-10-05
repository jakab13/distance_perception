import slab
import pathlib
import os
from os import listdir
from os.path import isfile, join

DIR = pathlib.Path().absolute()


def write_resampled(sound_type, resample_freq=48828):
    folder_path = DIR / 'experiment' / 'samples' / sound_type
    resampled_folder_path = folder_path.parent / str(sound_type + '_resampled')
    if not os.path.exists(resampled_folder_path):
        os.makedirs(resampled_folder_path)
    file_names = [f for f in listdir(folder_path)
                  if isfile(join(folder_path, f))
                  and not f.startswith('.')]
    for file_name in file_names:
        file_path = folder_path / file_name
        sound = slab.Binaural(file_path).resample(resample_freq)
        sound.write(resampled_folder_path/file_name)

write_resampled('bark')