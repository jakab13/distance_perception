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
    def __init__(self, participant_id, sound_type="pinknoise"):
        self.sound_type = sound_type
        self.sounds = load_sounds(self.sound_type)
        self.playback_direction = "random"
        self.record_response = False
        self.trials = None
        self.correct_total = 0
        self.deviant_freq = None
        self.participant_id = participant_id

    def get_distance_groups(self, playback_direction, scale_type):
        distance_groups = list(config['distance_groups'][scale_type])
        if playback_direction == 'away':
            distance_groups.sort()
            self.trials = distance_groups
        elif playback_direction == 'toward':
            distance_groups.sort(reverse=True)
            self.trials = distance_groups
        elif playback_direction == 'random':
            self.trials = None
        return distance_groups

    @staticmethod
    def crop_sound(sound, isi):
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

    def get_sound_from_group(self, group_number, scale_type):
        distances = config['distance_groups'][scale_type][group_number]
        distance = random.choice(distances)
        sounds = self.sounds[self.sound_type][distance]
        sound = random.choice(sounds)
        return sound

    def load_to_buffer(self, sound, isi=2.0):
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
            # response for deviant stimulus is reset to 0
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

    def run(self, playback_direction='random', scale_type='linear_10', record_response=False, n_reps=1, isi=1.5, level=75):
        self.record_response = record_response
        self.correct_total = 0
        distance_groups = self.get_distance_groups(playback_direction, scale_type=scale_type)
        seq = slab.Trialsequence(conditions=distance_groups, trials=self.trials, n_reps=n_reps,
                                 deviant_freq=self.deviant_freq)
        for distance_group in seq:
            if distance_group == 0:
                stimulus = self.sounds['deviant']
                stimulus.level = level - 10
            else:
                stimulus = self.get_sound_from_group(distance_group, scale_type=scale_type)
                stimulus.level = level
            print('Playing', self.sound_type, 'from group', distance_group)
            self.load_to_buffer(stimulus, isi)
            freefield.play()
            if not self.record_response:
                freefield.wait_to_finish_playing(proc="RP2", tag="playback")
            if self.record_response:
                self.collect_responses(seq)
        if self.record_response:
            seq.save_json("responses/" + "participant-" + self.participant_id +
                          "_training-" + self.sound_type + "_" + str(int(time.time())) + ".json")
            print("Saved participant responses")

    def play_control(self):
        control_sound = self.sounds[self.sound_type]['control']
        self.load_to_buffer(control_sound)
        freefield.play()

    def play_deviant(self):
        deviant_sound = self.sounds['deviant']
        deviant_sound.level -= 10
        self.load_to_buffer(deviant_sound)
        freefield.play()
