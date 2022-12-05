import slab
import pathlib
import os
import copy
import numpy
from scipy.signal import find_peaks
from os.path import join

SAMPLERATE = 44100
slab.Signal.set_default_samplerate(SAMPLERATE)
DIR = pathlib.Path(__file__).parent.parent.absolute()

filename_core = 'laughter'
folder_core = DIR / 'experiment' / 'samples' / 'VEs' / filename_core
simulated_folder_path = folder_core / 'simulated'
aligned_folder_path = folder_core / 'aligned'
a_weighted_folder_path = folder_core / 'a_weighted'

def abs_file_paths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            if not f.startswith('.'):
                yield pathlib.Path(join(dirpath, f))

def get_loudest_filename(folder_path, win_length=0.25):
    win_length = slab.Signal.in_samples(win_length, SAMPLERATE)
    file_paths = [f for f in abs_file_paths(folder_path)]
    loudest_filename = ''
    loudest_level = 0
    for file_path in file_paths:
        sound = slab.Binaural(file_path)
        sound.data = sound.data[0:win_length]
        level = numpy.average(sound.level)
        if level > loudest_level:
            loudest_filename = file_path
            loudest_level = level
    print(loudest_filename)
    return loudest_filename

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

def align_onset(sound, onset_length = 0.15):
    found_peaks = False
    peaks_left = []
    peaks_right = []
    peak_height = 1
    while found_peaks == False:
        peaks_left = find_peaks(sound.data[:, 0], height=peak_height)
        peaks_right = find_peaks(sound.data[:, 1], height=peak_height)
        if numpy.size(peaks_left[0]) == 0 or numpy.size(peaks_right[0]) == 0:
            peak_height -= 0.01
        else:
            found_peaks = True
    peak_idx = min(peaks_left[0][0], peaks_right[0][0])
    onset_length = slab.Signal.in_samples(onset_length, sound.samplerate)
    onset_idx = max(peak_idx - onset_length, 0)
    out = copy.deepcopy(sound)
    length = out.n_samples
    out.data[:length - onset_idx] = out.data[onset_idx:]
    filt = slab.Filter.band(frequency=50, kind='hp')
    out = filt.apply(out)
    print("aligned onset")
    return out

def generate_aligned_files(folder_path):
    file_paths = [f for f in abs_file_paths(folder_path)]
    aligned_folder_path = folder_path.parent / 'aligned'
    if not os.path.exists(aligned_folder_path):
        os.makedirs(aligned_folder_path)
    for file_path in file_paths:
        sound = slab.Binaural(file_path)
        aligned_sound = align_onset(sound)
        out_filename = 'A_' + file_path.name
        aligned_sound.write(aligned_folder_path / out_filename, normalise=False)

def equalise_a_weight(target, source, win_length=0.25):
    target_copy = copy.deepcopy(target)
    source_copy = copy.deepcopy(source)
    # target_copy = align_onset(target_copy)
    # source_copy = align_onset(source_copy)
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
    source_file_paths = [f for f in file_paths if "dis" in f.name]
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
    target_out_filename = 'AW_' + target_file_path[0].name
    target.write(a_weigthed_folder_path / target_out_filename, normalise=False)
    for file_name, sound in equalised_sources.items():
        source_out_filename = 'AW_' + file_name
        sound.write(a_weigthed_folder_path / source_out_filename, normalise=False)
    print('Done writing files')

# generate_aligned_files(simulated_folder_path)
generate_equalised_files(aligned_folder_path)

