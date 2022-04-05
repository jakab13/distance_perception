import slab
import pathlib
import os
import copy
import numpy
import scipy
from scipy import signal
from os.path import join

slab.Signal.set_default_samplerate(44100)
DIR = pathlib.Path(__file__).parent.parent.absolute()

filename_core = 'bark'
room_dimensions = '10-30-3'
sound_category = filename_core + '_' + 'room-' + room_dimensions
folder_core = DIR / 'experiment' / 'samples' / sound_category
simulated_folder_path = DIR / 'experiment' / 'samples' / sound_category / 'simulated'
aligned_folder_path = DIR / 'experiment' / 'samples' / sound_category / 'aligned'
a_weighted_folder_path = DIR / 'experiment' / 'samples' / sound_category / 'a_weighted'

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

def align_onset(sound):
    peaks_left = scipy.signal.find_peaks(sound.data[:, 0], height=0.001)
    peaks_right = scipy.signal.find_peaks(sound.data[:, 1], height=0.001)
    onset_idx = min(peaks_left[0][0], peaks_right[0][0])
    length = sound.n_samples
    sound.data[:length - onset_idx] = sound.data[onset_idx:]
    return sound

def generate_aligned_files(folder_path):
    file_paths = [f for f in abs_file_paths(folder_path)]
    for file_path in file_paths:
        sound = slab.Binaural(file_path)
        aligned_sound = align_onset(sound)
        out_filename = 'A_' + file_path.name
        aligned_folder_path = folder_path.parent / 'aligned'
        if not os.path.exists(aligned_folder_path):
            os.makedirs(aligned_folder_path)
        aligned_sound.write(aligned_folder_path / out_filename, normalise=False)

def equalise_a_weight(target, source, win_length=0.25):
    target_copy = copy.deepcopy(target)
    source_copy = copy.deepcopy(source)
    target_copy = align_onset(target_copy)
    source_copy = align_onset(source_copy)
    win_length = slab.Signal.in_samples(win_length, target_copy.samplerate)
    target_copy.data = target_copy.data[:win_length]
    source_copy.data = source_copy.data[:win_length]
    windowed_source_level = source_copy.level
    target_aw = target_copy.aweight()
    source_aw = source_copy.aweight()
    aw_level_diff = numpy.average(target_aw.level - source_aw.level)
    while aw_level_diff > 0:
        source_copy.level += 0.5
        source_aw = source_copy.aweight()
        aw_level_diff = numpy.average(target_aw.level - source_aw.level)
    if numpy.amax(source_copy.data) > 1:
        print('Calibration is clipping the output sound, please adjust target sound level')
    out = copy.deepcopy(source)
    out.level += source_copy.level - windowed_source_level
    return out

def generate_equalised_files(folder_path, win_length=0.25):
    file_paths = [f for f in abs_file_paths(folder_path)]
    target_file_path = [f for f in file_paths if "control" in f.name]
    source_file_paths = [f for f in file_paths if "dist" in f.name]
    target = slab.Binaural(target_file_path[0])
    equalised_sources = {}
    is_clipping = True
    while is_clipping:
        for source_file_path in source_file_paths:
            source = slab.Binaural(source_file_path)
            out = equalise_a_weight(target, source, win_length)
            if numpy.amax(out.data) > 1:
                is_clipping = True
                target.level -= 0.5
                break
            else:
                equalised_sources[source_file_path.name] = out
                is_clipping = False
    a_weigthed_folder_path = folder_path.parent / 'a_weighted'
    if not os.path.exists(a_weigthed_folder_path):
        os.makedirs(a_weigthed_folder_path)
    out_filename = 'AW_' + target_file_path[0].name
    target.write(a_weigthed_folder_path / out_filename, normalise=False)
    for file_name, sound in equalised_sources.items():
        out_filename = 'AW_' + file_name
        sound.write(a_weigthed_folder_path / out_filename, normalise=False)
    print('Done writing files')

# generate_aligned_files(simulated_folder_path)
generate_equalised_files(aligned_folder_path)
