import slab
import os
import pathlib
from os import listdir
from os.path import isfile, join


slab.set_default_samplerate(44100)
DIR = pathlib.Path(os.getcwd())

#for vocalist-2 and for vocalist-11 separately
folder_path = DIR / 'experiment' / 'samples' / 'vocal_effort' / 'vocalist-11' / 'pyloudnorm' / '300'


file_names = [pathlib.Path(folder_path / f) for f in listdir(folder_path)
              if isfile(join(folder_path, f))
              and not f.startswith('.')]

lengths = [0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025]

for file in file_names:
    sound = slab.Binaural(file)
    samplerate = sound.samplerate
    for length in lengths:
        length_in_samples = slab.Signal.in_samples(length, samplerate)
        sound.data = sound.data[:length_in_samples]
        length_string = str(int(length * 1000))
        folder_path_length = folder_path.parent / length_string
        if not os.path.exists(folder_path_length):
            os.makedirs(folder_path_length)
        sound.write(folder_path_length / str(file.stem + '_length-' + length_string + '.wav'))




