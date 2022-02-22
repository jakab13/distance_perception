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
import config
import load

slab.set_default_samplerate(44100)

config = config.get_config()
proc_list = config['proc_list']
freefield.initialize('dome', zbus=True, device=proc_list)
freefield.set_logger('WARNING')


class Training:
    def __init__(self, sound_type="pinknoise", room_dimensions='10-30-3', playback_direction='random'):
        self.playback_direction = playback_direction
        self.sound_type = sound_type
        self.room_dimensions = room_dimensions
        self.sounds = load.load_sounds(self.sound_type, self.room_dimensions)
        self.record_response = True
        self.jitter_distances = False
        self.trials = None
        self.correct_total = 0
        self.deviant_freq = None

    def get_distances(self):
        distances = [0.2, 0.4, 0.6, 0.8, 1, 1.6, 2.2, 3.0, 4, 5, 7, 10, 13, 16, 18]
        if self.playback_direction == 'away':
            distances.sort()
            self.trials = numpy.asarray([i + 1 for i in range(len(distances))])
            self.deviant_freq = None
        elif self.playback_direction == 'toward':
            distances.sort(reverse=True)
            self.trials = numpy.asarray([i + 1 for i in range(len(distances))])
            self.deviant_freq = None
        else:
            distances = [0.2, 2, 8, 18]
            self.jitter_distances = True
            self.deviant_freq = 0.05
        return distances

    @staticmethod
    def jitter_distance(distance):
        distance += random.uniform(-distance / 10, distance / 10)
        distance = 20 * round(distance / 20)
        return distance

    def crop_sound(self, sound, isi):
        isi = slab.Sound.in_samples(isi, sound.samplerate)
        # sound = copy.deepcopy(sound)
        if sound.n_samples < isi:
            silence_length = isi - sound.n_samples
            silence = slab.Sound.silence(duration=silence_length, samplerate=sound.samplerate)
            left = slab.Sound.sequence(sound.left, silence)
            right = slab.Sound.sequence(sound.right, silence)
            sound = slab.Binaural([left, right])
        else:
            sound.data = sound.data[: isi]
            sound = sound.ramp(duration=0.01)
        return sound

    def load_sound(self, sound, isi):
        isi = slab.Sound.in_samples(isi, sound.samplerate)
        isi = max(sound.n_samples, isi)
        freefield.write(tag="playbuflen", value=isi, processors="RP2")
        freefield.write(tag="data_l", value=sound.left.data.flatten(), processors="RP2")
        freefield.write(tag="data_r", value=sound.right.data.flatten(), processors="RP2")

    def collect_responses(self, seq):
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
        is_correct = True if response == seq.trials[seq.this_n] else False
        if is_correct:
            self.correct_total += 1
        seq.add_response({'solution': seq.trials[seq.this_n],
                          'response': response,
                          'isCorrect': is_correct,
                          'correct_total': self.correct_total})
        print('[Response ' + str(response) + ']', '(' + str(self.correct_total) + ')')

    def run(self, playback_direction='away', record_response=False, n_reps=1, isi=1.5):
        self.playback_direction = playback_direction
        distances = self.get_distances()
        seq = slab.Trialsequence(conditions=distances, trials=self.trials, n_reps=n_reps,
                                 deviant_freq=self.deviant_freq)
        for distance in seq:
            if distance == 0:
                stimulus = self.sounds['deviant']
            else:
                distance = int(distance * 100)
                if self.jitter_distances:
                    distance = self.jitter_distance(distance)
                stimulus = self.sounds[self.sound_type][self.room_dimensions][str(distance)]
            stimulus = self.crop_sound(stimulus, isi)
            print('[Distance ' + str(seq.trials[seq.this_n]) + ']',
                  self.sound_type, 'in room', self.room_dimensions,
                  'at', str(distance / 100) + 'm')
            # stimulus.play()
            self.load_sound(stimulus, isi)
            freefield.play()
            self.collect_responses(seq)
            if record_response:
                seq.save_json("responses.json", clobber=True)

    def play_control(self):
        control_sound = load.load_control(self.sound_type, self.room_dimensions)
        self.load_sound(control_sound)
        freefield.play()
        # control_sound.play()
