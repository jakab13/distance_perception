import slab
import pathlib
import os
import copy
import numpy
from scipy.signal import find_peaks
from os.path import join
import pyloudnorm as pyln

SAMPLERATE = 44100
slab.Signal.set_default_samplerate(SAMPLERATE)
DIR = pathlib.Path(__file__).parent.parent.absolute()

filename_core = 'laughter'
folder_core = DIR / 'experiment' / 'samples' / 'VEs' / filename_core
simulated_folder_path = folder_core / 'simulated'
aligned_folder_path = folder_core / 'aligned'
a_weighted_folder_path = folder_core / 'a_weighted'
normalised_folder_path = folder_core / 'normalised'


def abs_file_paths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            if not f.startswith('.'):
                yield pathlib.Path(join(dirpath, f))


def rewrite_loudest_file(folder_path, win_length=0.25):
    win_length = slab.Signal.in_samples(win_length, SAMPLERATE)
    file_paths = [f for f in abs_file_paths(folder_path)]
    has_control_file = next((True for file_path in file_paths if 'control' in file_path.name), False)
    if has_control_file:
        loudest_filename = ''
        loudest_level = 0
        for file_path in file_paths:
            sound = slab.Binaural(file_path)
            sound.data = sound.data[:win_length]
            sound = sound.aweight()
            level = numpy.average(sound.level)
            if level > loudest_level:
                loudest_filename = file_path
                loudest_level = level
        out = slab.Binaural(loudest_filename)
        out_file_name = loudest_filename.stem + '_control' + loudest_filename.suffix
        out.write(loudest_filename.parent / out_file_name)
        print("Wrote:", out_file_name)
    else:
        print("Control file already exists")


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


def align_onset(sound, onset_length=0.15):
    found_peaks = False
    peaks_left = []
    peaks_right = []
    peak_height = 1
    while not found_peaks:
        peaks_left = find_peaks(sound.data[:, 0], height=peak_height)
        peaks_right = find_peaks(sound.data[:, 1], height=peak_height)
        if numpy.size(peaks_left[0]) == 0 or numpy.size(peaks_right[0]) == 0:
            peak_height -= 0.05
        else:
            found_peaks = True
    peak_left_idx = numpy.argmax(peaks_left[1]['peak_heights'])
    peak_right_idx = numpy.argmax(peaks_right[1]['peak_heights'])
    peak_idx = min(peaks_left[0][peak_left_idx], peaks_right[0][peak_right_idx])
    onset_length = slab.Signal.in_samples(onset_length, SAMPLERATE)
    onset_idx = max(peak_idx - onset_length, 0)
    out = copy.deepcopy(sound)
    out.data = out.data[onset_idx:]
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
        aligned_folder_path = folder_path.parent / 'aligned'
        if not os.path.exists(aligned_folder_path):
            os.makedirs(aligned_folder_path)
        aligned_sound.write(aligned_folder_path / out_filename, normalise=False)

def generate_normalised_files(folder_path, duration=0.5, LUFS=-15.0):
    file_paths = [f for f in abs_file_paths(folder_path)]
    out_folder_path = folder_path.parent / 'normalised'
    if not os.path.exists(normalised_folder_path):
        os.makedirs(normalised_folder_path)
    duration = slab.Signal.in_samples(duration, SAMPLERATE)
    meter = pyln.Meter(SAMPLERATE, block_size=0.200)
    for file_path in file_paths:
        sound = slab.Binaural(file_path)
        sound.level = [numpy.average(sound.level), numpy.average(sound.level)]
        sound = align_onset(sound)
        sound.data = sound.data[:duration]
        sound = sound.ramp(duration=0.05)
        loudness = meter.integrated_loudness(sound.data)
        normalised_data = pyln.normalize.loudness(sound.data, loudness, LUFS)
        out_filename = 'N_' + file_path.name
        outfile = slab.Binaural(normalised_data)
        outfile.write(out_folder_path / out_filename, normalise=False)
    print("Done writing normalised files")


def play_normalised_seq(save=False):
    normalised_file_paths = [f for f in abs_file_paths(normalised_folder_path)]
    seq = slab.Trialsequence(conditions=normalised_file_paths)
    out = slab.Binaural.silence(duration=0)
    silence = slab.Binaural.silence(duration=1.0, samplerate=SAMPLERATE)
    for normalised_file_path in seq:
        sound = slab.Binaural(normalised_file_path)
        out = slab.Binaural.sequence(out, sound, silence)
        if not save:
            print(normalised_file_path.name)
            print(numpy.average(sound.spectral_feature(feature="flatness")) * 1000)
            print(numpy.average(sound.level))
            sound.play()
            input("ja?")
    if save:
        noise = slab.Binaural.pinknoise(duration=out.n_samples, level=40, kind="diotic")
        out = out + noise
        out.write(normalised_folder_path / 'stitched.wav', normalise=True)

