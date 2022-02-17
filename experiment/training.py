import slab
import pathlib
import os
import copy
import numpy
import scipy
from scipy import signal
from os import listdir
from os.path import isfile, join

slab.Signal.set_default_samplerate(44100)

DIR = pathlib.Path(os.getcwd())

filename_core = 'wow'
room_dimensions = '20-30-5'

file_category = filename_core + '_' + 'room-' + room_dimensions
a_weighted_filepath = DIR / 'experiment' / 'samples' / file_category / 'a_weighted'

a_weighted_sound_filenames = [f for f in sorted(listdir(a_weighted_filepath)) if isfile(join(a_weighted_filepath, f)) and not f.startswith('.')]

def play_sounds(n_reps, kind='random'):
    if kind == 'random':
        seq = slab.Trialsequence(a_weighted_sound_filenames, n_reps=n_reps)
    elif kind == 'away':
        seq = [filename for filename in a_weighted_sound_filenames]

    for filename in seq:
        isi = numpy.random.uniform(1.0, 1.0)
        stim = slab.Binaural(a_weighted_filepath / filename)
        isi = slab.Sound.in_samples(isi, stim.samplerate)
        if stim.nsamples < isi:
            silence_length = isi - stim.nsamples
            silence = slab.Binaural.silence(duration=silence_length, samplerate=stim.samplerate)
            stim = slab.Binaural.sequence(stim, silence)
            stim = stim.ramp(duration=0.01)
        else:
            stim.data = stim.data[: isi]
            stim = stim.ramp(duration=0.01)
        print('playing: ', filename)
        stim.play()


play_sounds(10, kind='random')
