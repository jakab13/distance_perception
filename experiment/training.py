import slab
import freefield
import pathlib
import os
import copy
import numpy
import scipy
from scipy import signal
from os import listdir
from os.path import isfile, join
import time
import random
from experiment.load import load_sounds, load_deviant, load_control

slab.set_default_samplerate(44100)

sound_type = 'pinknoise'
room_dimensions = '10-30-3'
distances = [0.2, 2, 8, 18]
# distances = [0.2, 0.4, 0.6, 0.8, 1, 1.6, 2.2, 3.0, 4, 5, 7, 10, 13, 16, 18]
# distances = numpy.arange(1, 20).tolist()
order = None
record_response = True
n_reps = 10

DIR = pathlib.Path(os.getcwd())  # path for sound and rcx files
proc_list = [['RP2', 'RP2',  DIR / 'experiment' / 'data' / 'bi_play_buf.rcx'],
             ['RX81', 'RX8',  DIR / 'experiment' / 'data' / 'play_buf.rcx'],
             ['RX82', 'RX8', DIR / 'experiment' / 'data' / 'play_buf.rcx']]
freefield.initialize('dome', zbus=True, device=proc_list)
freefield.set_logger('WARNING')


def play_sounds(sound_type="pinknoise", room_dimensions="5-30-5", distances=None, order=None, record_response=False, n_reps=1):
    trials = None
    randomise_distance = True
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
    seq = slab.Trialsequence(conditions=distances, trials=trials, n_reps=n_reps, deviant_freq=None)
    correct_total = 0
    counter = 0
    for distance in seq:
        if distance == 0:
            stim = deviant_sound
        else:
            distance = int(distance * 100)
            if randomise_distance:
                distance += random.uniform(-distance/10, distance/10)
                distance = 20 * round(distance/20)
            stim = loaded_sounds[sound_type][room_dimensions][str(distance)]

        isi = numpy.random.uniform(1.5, 1.5)
        isi = slab.Sound.in_samples(isi, stim.samplerate)

        if stim.n_samples < isi:
            silence_length = isi - stim.n_samples
            silence = slab.Sound.silence(duration=silence_length, samplerate=stim.samplerate)
            left = slab.Sound.sequence(stim.left, silence)
            right = slab.Sound.sequence(stim.right, silence)
            stim = slab.Binaural([left, right])
            stim = stim.ramp(duration=0.01)
        else:
            stim.data = stim.data[: isi]
            stim = stim.ramp(duration=0.01)
        print('[Distance ' + str(seq.trials[seq.this_n]) + ']',
              sound_type, 'in room', room_dimensions,
              'at', str(distance/100) + 'm')
        # stim.play()
        freefield.write(tag="playbuflen", value=isi, processors="RP2")
        freefield.write(tag="data_l", value=stim.left.data.flatten(), processors="RP2")
        freefield.write(tag="data_r", value=stim.right.data.flatten(), processors="RP2")
        freefield.play()
        response = None
        prev_response = 0
        while freefield.read(tag="playback", n_samples=1, processor="RP2"):
            curr_response = freefield.read(tag="response", processor="RP2")
            curr_response = int(curr_response)
            if curr_response > prev_response and curr_response != 0:
                response = int(numpy.log2(curr_response)) + 1
                if response == 5:
                    response = 0
            time.sleep(0.01)
            prev_response = curr_response
        counter += 1
        if record_response:
            is_correct = True if response == seq.trials[seq.this_n] else False
            if is_correct:
                correct_total += 1
            seq.add_response({'solution': seq.trials[seq.this_n],
                              'response': response,
                              'isCorrect': is_correct,
                              'correct_total': correct_total})
            print('[Response ' + str(response) + ']', '(' + str(correct_total) + '/' + str(counter) + ')')
    if record_response:
        seq.save_json("responses.json", clobber=True)


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