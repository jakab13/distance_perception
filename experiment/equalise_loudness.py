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
adjust = 12

DIR = pathlib.Path(os.getcwd())

filename_core = 'pinknoise'
room = 'room-5-30-3'

file_category = filename_core + '_' + room
simulated_filepath = DIR / 'experiment' / 'samples' / file_category / 'simulated'
aligned_filepath = DIR / 'experiment' / 'samples' / file_category / 'aligned'
a_weighted_filepath = DIR / 'experiment' / 'samples' / file_category / 'a_weighted'

simulated_sound_filenames = [f for f in listdir(simulated_filepath) if isfile(join(simulated_filepath, f))]
aligned_sound_filenames = [f for f in listdir(aligned_filepath) if isfile(join(aligned_filepath, f))]
a_weighted_sound_filenames = [f for f in listdir(a_weighted_filepath) if isfile(join(a_weighted_filepath, f))]

def write_pinknoises():
    pinknoise_list = slab.Precomputed(lambda: slab.Binaural.pinknoise(kind="dichotic", duration=0.25), n=10)
    i = 0
    for pinknoise in pinknoise_list:
        left = copy.deepcopy(pinknoise.left)
        right = copy.deepcopy(pinknoise.right)
        pinknoise.left = left.filter(frequency=20, kind="hp")
        pinknoise.right = right.filter(frequency=20, kind="hp")
        filename = 'pinknoise_' + str(i) + '.wav'
        pinknoise.write(filename)
        i += 1

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
    diff_count = 0
    while aw_level_diff > min_level_diff:
        source.level += 1
        diff_count += 1
        source_aw = source.aweight()
        aw_level_diff = numpy.average(target_aw.level - source_aw.level)
    out.level = source.level
    return out

def write_equalised_files(aligned_filepath):

    aligned_target_filenames = [s for s in aligned_sound_filenames if "m.wav" not in s]
    for aligned_target_filename in aligned_target_filenames:
        aligned_source_filenames = [s for s in aligned_sound_filenames if (aligned_target_filename[:-4] in s) & ("m.wav" in s)]
        for aligned_source_filename in aligned_source_filenames:
            aw_source = equalise_a_weight(aligned_filepath / aligned_target_filename, aligned_filepath / aligned_source_filename, 0.25)
            aligned_source_filename = 'AW_' + aligned_source_filename
            aw_source.write(a_weighted_filepath / aligned_source_filename, normalise=False)

        target = slab.Binaural(aligned_filepath / aligned_target_filename)
        target.level -= adjust/2
        aligned_target_filename = 'AW_' + aligned_target_filename
        target.write(a_weighted_filepath / aligned_target_filename, normalise=False)

def play_a_weighted_sounds(n_reps):
    seq = slab.Trialsequence(a_weighted_sound_filenames, kind="random_permutation", n_reps=10)
    for filename in seq:
        isi = numpy.random.uniform(1.0, 1.0)
        stim = slab.Binaural(a_weighted_filepath / filename)
        isi = slab.Sound.in_samples(isi, stim.samplerate)
        if stim.nsamples < isi:
            silence_length = isi - stim.nsamples
            silence = slab.Binaural.silence(duration=silence_length, samplerate=stim.samplerate)
            stim = slab.Binaural.sequence(stim, silence)
            stim = stim.ramp(duration=0.01)
        else:
            stim.data = stim.data[: isi]
            stim = stim.ramp(duration=0.01)
        print('playing: ', filename)
        stim.play()

# write_pinknoises()
# write_aligned_files(simulated_filepath, simulated_sound_filenames)
# write_equalised_files(aligned_filepath)
play_a_weighted_sounds(10)
