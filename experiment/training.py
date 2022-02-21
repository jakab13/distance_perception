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
from experiment.load import load_sounds, load_deviant, load_control

slab.set_default_samplerate(44100)

sound_type = 'plug'
room_dimensions = '10-30-3'
distances = [0.2, 2, 8, 18]
# distances = [0.2, 0.4, 0.6, 0.8, 1, 1.6, 2.2, 3.0, 4, 5, 7, 10, 13, 16, 18]
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
    deviant_sound = load_deviant()
    seq = slab.Trialsequence(conditions=distances, trials=trials, n_reps=n_reps, deviant_freq=0.1)
    correct_total = 0
    counter = 0
    for distance in seq:
        if distance == 0:
            stim = deviant_sound
        else:
            distance = int(distance * 100)
            distance += random.uniform(-distance/10, distance/10)
            distance = 20 * round(distance/20)
            stim = loaded_sounds[sound_type][room_dimensions][str(distance)]

        isi = numpy.random.uniform(1.5, 1.5)
        isi = slab.Sound.in_samples(isi, stim.samplerate)

        if stim.n_samples < isi:
            silence_length = isi - stim.n_samples
            silence = slab.Binaural.silence(duration=silence_length, samplerate=stim.samplerate)
            stim = slab.Binaural.sequence(stim, silence)
            stim = stim.ramp(duration=0.01)
        else:
            stim.data = stim.data[: isi]
            stim = stim.ramp(duration=0.01)
        print('playing:', sound_type, 'in room', room_dimensions, 'at', distance, 'cm', '[Group', seq.trials[seq.this_n], ']')
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


def play_control(sound_type=sound_type, room_dimensions=room_dimensions):
    control_sound = load_control(sound_type, room_dimensions)
    control_sound.play()


play_sounds(sound_type=sound_type,
            room_dimensions=room_dimensions,
            distances=distances,
            order=order,
            n_reps=n_reps,
            record_response=record_response)

# play_control(sound_type, room_dimensions)