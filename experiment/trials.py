import slab
import freefield
import copy
import pathlib
import numpy
import time
import random
import string
from experiment.config import get_config
from experiment.load import load_sounds

slab.set_default_samplerate(44100)
DIR = pathlib.Path(__file__).parent.absolute()


class Experiment:
    _participant_id = None

    def initialise(self):
        config = get_config()
        proc_list = config['proc_list']
        freefield.initialize('dome', zbus=True, device=proc_list)
        freefield.set_logger('WARNING')

    def generate_id(self):
        characters = string.ascii_lowercase + string.digits
        self._participant_id = ''.join(random.choice(characters) for i in range(6))
        print("Participant ID is:", self._participant_id)

    @property
    def participant_id(self):
        if self._participant_id is None:
            self.generate_id()
        return self._participant_id


class Trials:
    def __init__(self, participant_id, sound_type="pinknoise"):
        self.sound_type = sound_type + '_resampled'
        self.sounds = load_sounds(self.sound_type)
        self.loudnesses = self.get_loudnesses()
        self.trials = None
        self.correct_total = 0
        self.participant_id = participant_id
        self.config = get_config()

    def load_config(self):
        self.config = get_config()

    def get_loudnesses(self):
        loudnesses = {}
        for dist, ids in self.sounds[self.sound_type].items():
            if type(dist) is int and dist is not 0:
                for id, sound in self.sounds[self.sound_type][dist].items():
                    loudnesses[dist] = sound.level
        loudness_min = min([numpy.average(level) for id, level in loudnesses.items()])
        loudness_max = max([numpy.average(level) for id, level in loudnesses.items()])
        return [loudness_min, loudness_max]

    def get_distance_groups(self, playback_direction, scale_type):
        distance_groups = list(self.config['distance_groups'][scale_type])
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

    def get_sound_from_group(self, group_number, scale_type='vocal_effort', sound_id='random'):
        if group_number == 0:
            distance = 0
            sound_id = 0
        else:
            distances = self.config['distance_groups'][scale_type][group_number]
            distance = random.choice(distances)
        if sound_id == 'random':
            id_choices = self.config['selected_USO_IDs']
            sound_id = random.choice(id_choices)
        sound = self.sounds[self.sound_type][distance][sound_id]
        return sound, distance

    def load_to_buffer(self, sound, isi=1.0):
        out = self.crop_sound(sound, isi)
        isi = slab.Sound.in_samples(isi, 48828)
        freefield.write(tag="playbuflen", value=isi, processors="RP2")
        freefield.write(tag="data_l", value=out.left.data.flatten(), processors="RP2")
        freefield.write(tag="data_r", value=out.right.data.flatten(), processors="RP2")

    def collect_responses(self, seq, results_file, sound_id):
        response = None
        reaction_time = None
        start_time = time.time()
        while not freefield.read(tag="response", processor="RP2"):
            time.sleep(0.01)
        curr_response = int(freefield.read(tag="response", processor="RP2"))
        if curr_response != 0:
            reaction_time = int(round(time.time() - start_time, 3) * 1000)
            response = int(numpy.log2(curr_response))
            # response for deviant stimulus is reset to 0
        is_correct = response == seq.trials[seq.this_n]
        if is_correct:
            self.correct_total += 1
        results_file.write(sound_id, tag='sound_id')
        results_file.write(seq.trials[seq.this_n], tag='solution')
        results_file.write(response, tag='response')
        results_file.write(is_correct, tag='is_correct')
        results_file.write(self.correct_total, tag='correct_total')
        results_file.write(reaction_time, tag='reaction_time')
        print('[Response ' + str(response) + ']',
              '(Correct ' + str(self.correct_total) + '/' + str(seq.n_trials) + ')')
        while freefield.read(tag="playback", n_samples=1, processor="RP2"):
            time.sleep(0.01)

    def button_trig(self, trig_value, seq, results_file, sound_id):
        prev_response = 0
        while freefield.read(tag="playback", n_samples=1, processor="RP2"):
            curr_response = freefield.read(tag="response", processor="RP2")
            if curr_response > prev_response:
                print("button was pressed")
                response = int(numpy.log2(curr_response))
                solution = seq.trials[seq.this_n - 1]
                is_correct = response == solution
                if is_correct:
                    self.correct_total += 1
                results_file.write(sound_id, tag='sound_id')
                results_file.write(solution, tag='solution')
                results_file.write(response, tag='response')
                results_file.write(is_correct, tag='is_correct')
                results_file.write(self.correct_total, tag='correct_total')
                freefield.write(tag='trigcode', value=trig_value, processors='RX82')
                print("trigcode was set to:", trig_value)
                freefield.play(proc='RX82')
            time.sleep(0.01)
            prev_response = curr_response

    def run(self, stage='training', playback_direction='random', scale_type='USO_scale', sound_id='random',
            record_response=False, n_reps=1, seq_length=20, isi=1.0, level=65):
        results_folder = DIR / 'results' / 'USOs'
        results_file = slab.ResultsFile(subject=self.participant_id, folder=results_folder)
        results_file.write(stage, tag='stage')
        results_file.write(self.sound_type, tag='sound_type')
        self.correct_total = 0
        self.load_config()
        sound_id = sound_id if self.sound_type == 'USOs_resampled' else 0
        scale_type = 'vocal_effort' if 'vocalist' in self.sound_type else scale_type
        deviant_freq = 0.1 if stage == 'experiment' else None
        distance_groups = self.get_distance_groups(playback_direction, scale_type=scale_type)
        distance_seq = slab.Trialsequence(conditions=distance_groups, trials=self.trials, n_reps=n_reps,
                                 deviant_freq=deviant_freq)
        uso_seq = slab.Trialsequence(conditions=self.config['selected_USO_IDs'], kind='infinite')
        for distance_group in distance_seq:
            if self.sound_type == 'USOs_resampled' and stage == 'experiment':
                sound_id = uso_seq.get_future_trial((int(distance_seq.this_n / seq_length)) % len(self.config['selected_USO_IDs']) + 1)
            stimulus, distance = self.get_sound_from_group(distance_group, scale_type=scale_type, sound_id=sound_id)
            stimulus.level = level
            loudness_avg = (self.loudnesses[0] + self.loudnesses[1]) / 2
            stimulus.level = numpy.interp(level, [self.loudnesses[0], self.loudnesses[1]], [loudness_avg - 1.5, loudness_avg + 1.5])
            print('Playing from distance', distance_group, '(' + str(distance_seq.this_n + 1) + '/' + str(distance_seq.n_trials) + ')')
            self.load_to_buffer(stimulus, isi=isi)
            trig_value = distance_group if distance_group != 0 else 6
            freefield.write(tag='trigcode', value=trig_value, processors='RX82')
            freefield.play()
            distance_seq.add_response({'sound_id': sound_id})
            if stage == 'experiment':
                self.button_trig(7, distance_seq, results_file, sound_id)
            if not record_response:
                freefield.wait_to_finish_playing(proc="RP2", tag="playback")
            if record_response:
                self.collect_responses(distance_seq, results_file, sound_id)
        results_file.write(distance_seq, tag='sequence')
        print("Saved participant responses")

    def play_control(self, sound_id='random', level=65, isi=1.0):
        control_sounds = self.sounds[self.sound_type]['controls']
        if sound_id == 'random':
            control_sound = random.choice(control_sounds)
        else:
            control_sound = control_sounds[sound_id]
        control_sound.level = level
        self.load_to_buffer(control_sound, isi)
        freefield.play()

    def play_deviant(self):
        deviant_sound = self.sounds[self.sound_type][0].random_choice()[0]
        self.load_to_buffer(deviant_sound)
        freefield.play()
        freefield.wait_to_finish_playing()

    def run_control(self, sound_id='random', n_reps=1, isi=0.7):
        seq = slab.Trialsequence(conditions=[0, 1, 2, 3, 4], n_reps=n_reps)
        for condition in seq:
            trig_value = 1
            freefield.write(tag='trigcode', value=trig_value, processors='RX82')
            self.play_control(sound_id=sound_id, isi=isi)
