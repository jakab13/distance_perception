import slab
import pathlib
import os
import copy
import numpy
import scipy
from scipy import signal
from os import listdir
from os.path import isfile, join
import random
from experiment.load import load_sounds

slab.Signal.set_default_samplerate(44100)

sound_type = 'waterdrop'
room_dimensions = '10-30-3'
distances = [0.2, 1, 4, 16]
# distances = numpy.arange(1, 20).tolist()
order = None
record_response = True
n_reps = 10


def play_sounds(sound_type="pinknoise", room_dimensions="5-30-5", distances=None, order=None, record_response=False, n_reps=1):
    trials = None
    if order == 'away':
        distances.sort()
        n_reps = 1
        trials = numpy.asarray([i + 1 for i in range(len(distances))])
    elif order == 'toward':
        distances.sort(reverse=True)
        n_reps = 1
        trials = numpy.asarray([i + 1 for i in range(len(distances))])
    loaded_sounds = load_sounds(sound_type, room_dimensions)
    seq = slab.Trialsequence(conditions=distances, trials=trials, n_reps=n_reps)
    correct_total = 0
    counter = 0
    for distance in seq:
        distance = str(int(distance * 100))
        isi = numpy.random.uniform(1.5, 1.5)
        stim = loaded_sounds[sound_type][room_dimensions][distance]
        isi = slab.Sound.in_samples(isi, stim.samplerate)
        if stim.nsamples < isi:
            silence_length = isi - stim.nsamples
            silence = slab.Binaural.silence(duration=silence_length, samplerate=stim.samplerate)
            stim = slab.Binaural.sequence(stim, silence)
            stim = stim.ramp(duration=0.01)
        else:
            stim.data = stim.data[: isi]
            stim = stim.ramp(duration=0.01)
        print('playing:', sound_type, 'in room', room_dimensions, 'at', distance, 'cm')
        stim.play()
        counter += 1
        if record_response:
            response = random.choice([1, 2, 3, 4])
            is_correct = True if response == seq.trials[seq.this_n] else False
            if is_correct:
                correct_total += 1
            seq.add_response({'solution': seq.trials[seq.this_n], 'response': response, 'isCorrect': is_correct})
            if seq.finished:
                seq.add_response({'correct_total': correct_total})
            print('Correct', correct_total, '/', counter)
    if record_response:
        seq.save_json("responses.json")


play_sounds(sound_type=sound_type,
            room_dimensions=room_dimensions,
            distances=distances,
            order=order,
            n_reps=n_reps,
            record_response=record_response)
