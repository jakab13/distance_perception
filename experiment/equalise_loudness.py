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

setup_table = {
    # "bark": {
    #     "window": 0.4,
    #     "adjust": 8
    # },
    # "beam": {
    #     "window": 0.2,
    #     "adjust": 10
    # },
    # "bum": {
    #     "window": 0.19,
    #     "adjust": 8
    # },
    # "chirp": {
    #     "window": 0.3,
    #     "adjust": 50
    # },
    # "coin_beep": {
    #     "window": 0.15,
    #     "adjust": 25
    # },
    # "drip": {
    #     "window": 0.1,
    #     "adjust": 15
    # },
    # "dunk": {
    #     "window": 0.3,
    #     "adjust": 20
    # },
    # "glass": {
    #     "window": 0.15,
    #     "adjust": 18
    # },
    # "lock": {
    #     "window": 0.15,
    #     "adjust": 15
    # },
    # "pinknoise": {
    #     "window": 0.25,
    #     "adjust": 13
    # },
    "pinknoise_ramped": {
        "window": 0.25,
        "adjust": 15
    },
    # "plug": {
    #     "window": 0.1,
    #     "adjust": 12
    # },
    # "sneeze": {
    #     "window": 0.4,
    #     "adjust": 10
    # },
    # "sonar_echo": {
    #     "window": 0.5,
    #     "adjust": 15
    # },
    # "stab": {
    #     "window": 0.4,
    #     "adjust": 20
    # },
    # "waterdrop": {
    #     "window": 0.15,
    #     "adjust": 18
    # },
    # "whisper": {
    #     "window": 0.4,
    #     "adjust": 10
    # },
    # "wow": {
    #     "window": 0.4,
    #     "adjust": 5
    # }
}

room_dimensions = '10-30-3'


def write_pinknoises(envelope):
    pinknoise_list = slab.Precomputed(lambda: slab.Binaural.pinknoise(kind="dichotic", duration=0.25), n=1)
    i = 0
    for pinknoise in pinknoise_list:
        left = copy.deepcopy(pinknoise.left)
        right = copy.deepcopy(pinknoise.right)
        pinknoise.left = left.filter(frequency=20, kind="hp")
        pinknoise.right = right.filter(frequency=20, kind="hp")
        pinknoise = pinknoise.envelope(apply_envelope=envelope)
        filename = 'pinknoise_' + str(i) + '.wav'
        pinknoise.write(filename)
        i += 1

# plug_sound = slab.Binaural(DIR / 'experiment' / 'samples' / 'plug_room-10-30-3' / 'a_weighted'/ 'AW_A_plug_room-10-30-3_control.wav')
# bum_sound = slab.Binaural(DIR / 'experiment' / 'samples' / 'bum_room-10-30-3' / 'a_weighted'/ 'AW_A_bum_room-10-30-3_control.wav')
# env = plug_sound.envelope()
# env = bum_sound.envelope()
# env = env[:,0]
# decay_curve = [numpy.exp(-i * 1) for i in range(10)]
# sound = slab.Binaural.pinknoise(kind="dichotic", duration=0.25)
# impulse = sound.envelope(apply_envelope=env)
# impulse.waveform()
# plug_sound.waveform()
# impulse.write('pinknoise.wav')
# write_pinknoises(env)

for filename_core in setup_table:
    adjust = setup_table[filename_core]["adjust"]

    file_category = filename_core + '_' + 'room-' + room_dimensions
    simulated_filepath = DIR / 'experiment' / 'samples' / file_category / 'simulated'
    aligned_filepath = DIR / 'experiment' / 'samples' / file_category / 'aligned'
    a_weighted_filepath = DIR / 'experiment' / 'samples' / file_category / 'a_weighted'

    if not os.path.exists(aligned_filepath):
        os.makedirs(aligned_filepath)

    if not os.path.exists(a_weighted_filepath):
        os.makedirs(a_weighted_filepath)

    simulated_sound_filenames = [f for f in listdir(simulated_filepath) if isfile(join(simulated_filepath, f)) and not f.startswith('.')]
    aligned_sound_filenames = [f for f in listdir(aligned_filepath) if isfile(join(aligned_filepath, f)) and not f.startswith('.')]
    a_weighted_sound_filenames = [f for f in listdir(a_weighted_filepath) if isfile(join(a_weighted_filepath, f)) and not f.startswith('.')]

    def align_onset(filename):
        sound = slab.Binaural(filename)
        peaks_left = scipy.signal.find_peaks(sound.data[:, 0], height=0.001)
        peaks_right = scipy.signal.find_peaks(sound.data[:, 1], height=0.001)
        onset_idx = min(peaks_left[0][0], peaks_right[0][0])
        for chan_num in range(sound.n_channels):
            length = sound.n_samples
            sound.data[:length - onset_idx, chan_num] = sound.data[onset_idx:, chan_num]
        return sound

    def write_aligned_files(filepath, filenames):
        for filename in filenames:
            aligned_sound = align_onset(filepath / filename)
            out_filename = 'A_' + filename
            aligned_sound.write(aligned_filepath / out_filename, normalise=False)

    def equalise_a_weight(target_filename, source_filename, length):
        target = slab.Binaural(target_filename)
        source = slab.Binaural(source_filename)
        out = copy.deepcopy(source)
        length_in_samples = slab.Signal.in_samples(length, target.samplerate)
        target.data = target.data[:length_in_samples]
        source.data = source.data[:length_in_samples]
        target.level -= adjust
        source.level -= adjust
        target_aw = target.aweight()
        source_aw = source.aweight()
        aw_level_diff = numpy.average(target_aw.level - source_aw.level)
        min_level_diff = 0.01
        while aw_level_diff > min_level_diff:
            source.level += 1
            source_aw = source.aweight()
            aw_level_diff = numpy.average(target_aw.level - source_aw.level)
        out.level = source.level
        if numpy.amax(out.data) > 1:
            raise ValueError("too loud")
        return out

    def write_equalised_files(aligned_filepath):

        aligned_target_filenames = [s for s in aligned_sound_filenames if "dist" not in s]
        for aligned_target_filename in aligned_target_filenames:
            aligned_source_filenames = [s for s in aligned_sound_filenames if "dist" in s]
            for aligned_source_filename in aligned_source_filenames:
                window_length = setup_table[filename_core]["window"]
                aw_source = equalise_a_weight(aligned_filepath / aligned_target_filename, aligned_filepath / aligned_source_filename, window_length)
                output_filename = 'AW_' + aligned_source_filename
                aw_source.write(a_weighted_filepath / output_filename, normalise=False)
                print('writing ', output_filename)

            target = slab.Binaural(aligned_filepath / aligned_target_filename)
            output_filename = 'AW_' + aligned_target_filename
            target.write(a_weighted_filepath / output_filename, normalise=False)

    # write_aligned_files(simulated_filepath, simulated_sound_filenames)
    write_equalised_files(aligned_filepath)
