import slab
import pathlib
import os
import copy
import numpy
import scipy
from scipy import signal
from os import listdir
from os.path import isfile, join, abspath

slab.Signal.set_default_samplerate(44100)

DIR = pathlib.Path(__file__).parent.parent.absolute()

setup_table = {
    "bark": {
        "window": 0.4,
        "adjust": 8
    },
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
    "dunk": {
        "window": 0.3,
        "adjust": 20
    },
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

def abs_file_paths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            if not f.startswith('.'):
                yield pathlib.Path(join(dirpath, f))

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

filename_core = 'pinknoise_ramped'
sound_category = filename_core + '_' + 'room-' + room_dimensions
folder_core = DIR / 'experiment' / 'samples' / sound_category
simulated_folder_path = DIR / 'experiment' / 'samples' / sound_category / 'simulated'
aligned_folder_path = DIR / 'experiment' / 'samples' / sound_category / 'aligned'
a_weighted_folder_path = DIR / 'experiment' / 'samples' / sound_category / 'a_weighted'

adjust = setup_table[filename_core]["adjust"]

if not os.path.exists(a_weighted_folder_path):
    os.makedirs(a_weighted_folder_path)

a_weighted_sound_filenames = [f for f in listdir(a_weighted_folder_path) if isfile(abspath(join(a_weighted_folder_path, f))) and not f.startswith('.')]

def align_onset(file_path):
    sound = slab.Binaural(file_path)
    peaks_left = scipy.signal.find_peaks(sound.data[:, 0], height=0.001)
    peaks_right = scipy.signal.find_peaks(sound.data[:, 1], height=0.001)
    onset_idx = min(peaks_left[0][0], peaks_right[0][0])
    length = sound.n_samples
    sound.data[:length - onset_idx] = sound.data[onset_idx:]
    return sound

def write_aligned_files(folder_path):
    file_paths = [f for f in abs_file_paths(folder_path)]
    for file_path in file_paths:
        aligned_sound = align_onset(file_path)
        out_filename = 'A_' + file_path.name
        aligned_folder_path = folder_path.parent / 'aligned'
        if not os.path.exists(aligned_folder_path):
            os.makedirs(aligned_folder_path)
        aligned_sound.write(aligned_folder_path / out_filename, normalise=False)

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

write_aligned_files(simulated_folder_path)
# write_equalised_files(aligned_filepath)
