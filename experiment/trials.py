import slab
import freefield
import copy
import numpy
import time
import random
from experiment.config import get_config
from experiment.load import load_sounds

slab.set_default_samplerate(44100)
config = get_config()
proc_list = config['proc_list']
freefield.initialize('dome', zbus=True, device=proc_list)
freefield.set_logger('WARNING')


class Trials:
    def __init__(self, participant_id, sound_type="pinknoise", room_dimensions='10-30-3'):
        self.sound_type = sound_type
        self.room_dimensions = room_dimensions
        self.sounds = load_sounds(self.sound_type, self.room_dimensions)
        self.playback_direction = "random"
        self.record_response = True
        self.jitter_distances = False
        self.trials = None
        self.correct_total = 0
        self.deviant_freq = None
        self.participant_id = participant_id

    def get_distances(self, playback_direction):
        distances = config['distances']['detailed']
        if playback_direction == 'away':
            distances.sort()
            self.trials = distances
            self.deviant_freq = None
            self.jitter_distances = False
        elif playback_direction == 'toward':
            distances.sort(reverse=True)
            self.trials = distances
            self.deviant_freq = None
            self.jitter_distances = False
        elif playback_direction == 'random':
            distances = config['distances']['sparse']
            self.jitter_distances = True
            self.deviant_freq = None
            self.record_response = True
            self.trials = None
        return distances

    @staticmethod
    def jitter_distance(distance):
        distance += random.uniform(-distance / 10, distance / 10)
        distance = 20 * round(distance / 20)
        return distance

    def crop_sound(self, sound, isi):
        isi = slab.Sound.in_samples(isi, sound.samplerate)
        out = copy.deepcopy(sound)
        if sound.n_samples < isi:
            silence_length = isi - sound.n_samples
            silence = slab.Sound.silence(duration=silence_length, samplerate=sound.samplerate)
            left = slab.Sound.sequence(sound.left, silence)
            right = slab.Sound.sequence(sound.right, silence)
            out = slab.Binaural([left, right])
        else:
            out.data = sound.data[: isi]
        out = out.ramp(duration=0.01)
        return out

    def load_sound(self, sound, isi=2.0):
        out = self.crop_sound(sound, isi)
        isi = slab.Sound.in_samples(isi, out.samplerate)
        isi = max(out.n_samples, isi)
        freefield.write(tag="playbuflen", value=isi, processors="RP2")
        freefield.write(tag="data_l", value=out.left.data.flatten(), processors="RP2")
        freefield.write(tag="data_r", value=out.right.data.flatten(), processors="RP2")

    def collect_responses(self, seq):
        response = None
        reaction_time = None
        start_time = time.time()
        while not freefield.read(tag="response", processor="RP2"):
            time.sleep(0.01)
        curr_response = int(freefield.read(tag="response", processor="RP2"))
        if curr_response != 0:
            reaction_time = int(round(time.time() - start_time, 3) * 1000)
            response = int(numpy.log2(curr_response)) + 1
            if response == 5:
                response = 0
        is_correct = response == seq.trials[seq.this_n]
        if is_correct:
            self.correct_total += 1
        seq.add_response({'solution': seq.trials[seq.this_n],
                          'response': response,
                          'isCorrect': is_correct,
                          'correct_total': self.correct_total,
                          'rt': reaction_time})
        print('[Response ' + str(response) + ']',
              '(Correct ' + str(self.correct_total) + '/' + str(seq.this_n + 1) + ')')
        while freefield.read(tag="playback", n_samples=1, processor="RP2"):
            time.sleep(0.01)

    def run(self, playback_direction='random', record_response=False, n_reps=1, isi=1.5, level=75):
        self.record_response = record_response
        self.correct_total = 0
        distances = self.get_distances(playback_direction)
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
            stimulus.level = level
            print('Playing', self.sound_type, 'in room', self.room_dimensions,
                  'at', str(distance / 100) + 'm')
            self.load_sound(stimulus, isi)
            freefield.play()
            if not self.record_response:
                freefield.wait_to_finish_playing(proc="RP2", tag="playback")
            # stimulus.play()
            if self.record_response:
                self.collect_responses(seq)
        if self.record_response:
            print("anyad")
            seq.save_json("responses/" + "participant-" + self.participant_id +
                          "_training-" + self.sound_type + "_" + str(int(time.time())) + ".json")

    def play_control(self):
        control_sound = self.sounds[self.sound_type][self.room_dimensions]['control']
        self.load_sound(control_sound)
        freefield.play()
        # control_sound.play()

    def play_deviant(self):
        deviant_sound = self.sounds['deviant']
        deviant_sound.level -= 10
        self.load_sound(deviant_sound)
        freefield.play()
        # control_sound.play()
