import slab
import pathlib
import os
import numpy
import copy
from scipy.signal import find_peaks
from os.path import join
import pyloudnorm as pyln
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pprint import pprint

SAMPLERATE = 44100
slab.Signal.set_default_samplerate(SAMPLERATE)
DIR = pathlib.Path(os.getcwd())

filename_core = 'vocoded-11-reconstructed'
folder_core = DIR / 'experiment' / 'samples' / 'VEs'/ filename_core
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
    if peak_idx < onset_length:
        silence_length = onset_length - peak_idx
        silence = slab.Binaural.silence(duration=int(silence_length))
        out = copy.deepcopy(sound)
        out = slab.Binaural.sequence(silence, out)
    else:
        onset_idx = max(peak_idx - onset_length, 0)
        out = copy.deepcopy(sound)
        out.data = out.data[onset_idx:]
    return out


def generate_aligned_files(folder_path, onset_length=0.15):
    file_paths = [f for f in abs_file_paths(folder_path)]
    aligned_folder_path = folder_path.parent / 'aligned'
    if not os.path.exists(aligned_folder_path):
        os.makedirs(aligned_folder_path)
    for file_path in file_paths:
        sound = slab.Binaural(file_path)
        aligned_sound = align_onset(sound, onset_length)
        aligned_sound = aligned_sound.ramp(duration=0.05)
        out_filename = 'A_' + file_path.name
        aligned_folder_path = folder_path.parent / 'aligned'
        if not os.path.exists(aligned_folder_path):
            os.makedirs(aligned_folder_path)
        aligned_sound.write(aligned_folder_path / out_filename, normalise=False)

def generate_normalised_files(folder_path, duration=0.5, LUFS=-25.0, type="pyloudnorm"):
    file_paths = [f for f in abs_file_paths(folder_path)]
    out_folder_path = folder_path.parent / 'normalised' / type
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)
    duration = slab.Signal.in_samples(duration, SAMPLERATE)
    meter = pyln.Meter(SAMPLERATE, block_size=0.200)
    for file_path in file_paths:
        if type == "pyloudnorm":
            sound = slab.Binaural(file_path)
            sound.level = [numpy.average(sound.level), numpy.average(sound.level)]
            # sound = align_onset(sound)
            snippet = copy.deepcopy(sound)
            snippet.data = snippet.data[:duration]
            snippet = snippet.ramp(duration=0.05)
            loudness = meter.integrated_loudness(snippet.data)
            normalised_snippet = pyln.normalize.loudness(snippet.data, loudness, LUFS)
            normalised_loudness = slab.Binaural(normalised_snippet).level
            sound.level = normalised_loudness
            out_filename = 'N_pyloudnorm_' + file_path.name
            sound.write(out_folder_path / out_filename, normalise=False)
        elif type == "pydub":
            sound = AudioSegment.from_file(file_path, "wav")
            snippet = sound[:int(duration*1000)]
            target_dBFS = LUFS
            change_in_dBFS = target_dBFS - snippet.dBFS
            normalized_sound = sound.apply_gain(change_in_dBFS)
            normalized_sound.export(out_folder_path / str('N_pydub_' + file_path.name), format="wav")
    print("Done writing normalised files")


def play_normalised_seq(folder_path, duration=1.0, save=False):
    file_paths = [f for f in abs_file_paths(folder_path)]
    seq = slab.Trialsequence(conditions=file_paths)
    out = slab.Binaural.silence(duration=0)
    silence = slab.Binaural.silence(duration=0.001, samplerate=SAMPLERATE)
    for file_path in seq:
        sound = slab.Binaural(file_path)
        sound.data = sound.data[:slab.Signal.in_samples(duration, SAMPLERATE)]
        sound = sound.ramp()
        out = slab.Binaural.sequence(out, sound, silence)
        if not save:
            print(file_path.name)
            sound.play()
            input("ja?")
    if save:
        noise = slab.Binaural.pinknoise(duration=out.n_samples, level=40, kind="diotic")
        out = out + noise
        out.write(folder_path / 'stitched.wav', normalise=True)

def plot_ve_envs(folder_path):
    filename_core = 'vocoded-all-reconstructed-test'
    folder_core = DIR / 'experiment' / 'samples' / 'VEs' / filename_core
    folder_path = folder_core
    file_names = sorted(folder_path.glob('*.wav'))
    results = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    single_envs = dict()
    for file_name in file_names:
        distance = file_name.name[file_name.name.find('dist-') + len('dist-'):file_name.name.rfind('_try')]
        sig = slab.Sound(file_name)
        res = sig.envelope()
        single_envs[file_name.name] = res
        print(distance)
        results[int(distance)][str(file_name.stem)] = res
    envs = [None] * 5
    for i in range(5):
        envs[i] = numpy.mean([results[i+1][key].data for id, key in enumerate(results[i+1])], axis=0)
    for i, key in enumerate(envs):
        plt.plot(numpy.mean(envs[i], axis=1), label="Vocal Effort {}".format(i+1))
    plt.legend()
    plt.show()
    plt.title("{} - Average sound energy".format(folder_path.parent.parent.name + "_" + folder_path.name))
    plt.xlabel('Time (s)')
    plt.ylabel('RMS (normalised)')
    plt.plot(sound_energy)
    plt.fill_between(numpy.arange(0, len(envs[0])), sound_energy, alpha=0.2)
    plt.xlim((-0.2 * 44100, 0.7 * 44100))
    plt.xticks([-0.1 * 44100, 0, 0.1 * 44100, 0.2 * 44100, 0.3 * 44100, 0.4 * 44100, 0.5 * 44100, 0.6 * 44100],
               [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    plt.yticks(numpy.linspace(0, numpy.amax(sound_energy), 5), numpy.linspace(0, 1, 5))

    sound_energy = numpy.mean(numpy.mean(envs, axis=0), axis=1)
    plt.title("Average sound energy - 500ms")
    plt.xlabel('Time (s)')
    plt.ylabel('RMS (normalised)')
    plt.plot(sound_energy)
    plt.fill_between(numpy.arange(0, len(envs[0])), sound_energy, alpha=0.2)
    plt.xlim((-0.2 * 44100, 0.7 * 44100))
    plt.xticks([-0.1 * 44100, 0, 0.1 * 44100, 0.2 * 44100, 0.3 * 44100, 0.4 * 44100, 0.5 * 44100, 0.6 * 44100],
               [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    plt.yticks(numpy.linspace(0, numpy.amax(sound_energy), 5), numpy.linspace(0, 1, 5))

    plt.legend()
    plt.show()

distance_groups = {
    1: [0.18, 0.19, 0.2, 0.21, 0.22],
    2: [1.62, 1.71, 1.8, 1.89, 1.98],
    3: [4.5, 4.75, 5, 5.25, 5.5, ],
    4: [10.8, 11.4, 12, 12.6, 13.2],
    5: [22.5, 23.75, 25, 26.25, 27.5]
}

distances = [0.18, 0.19, 0.2, 0.21, 0.22, 1.62, 1.71, 1.8, 1.89, 1.98,
             4.5, 4.75, 5, 5.25, 5.5,
             10.8, 11.4, 12, 12.6, 13.2,
             22.5, 23.75, 25, 26.25, 27.5];

def plot_pn_envs(folder_path, distance_groups):
    file_names = sorted(folder_path.glob('*.wav'))
    results = {}
    # single_envs = dict()
    for file_name in file_names:
        distance = file_name.name[file_name.name.find('dist-') + len('dist-'):file_name.name.rfind('.wav')]
        sig = slab.Sound(file_name)
        res = sig.envelope()
        # single_envs[file_name.name] = res
        results[int(distance)] = res
    envs = [None] * 5
    for i, distance_group in enumerate(distance_groups):
        envs[i] = numpy.mean([results[distance].data[:int(44100*0.5)]
                                           for distance in distance_groups[distance_group]], axis=0)
    for i, key in enumerate(envs):
        plt.plot(numpy.mean(envs[i], axis=1), label="Distance group " + str(i+1))
                                                    # str(numpy.average(distance_groups[i+1])/100) + "m")
    plt.title("{} - Average sound energy".format(folder_path.parent.parent.name + "_" + folder_path.name))
    plt.legend()
    plt.show()

for type in ["pydub", "pyloudnorm"]:
    folder_path = normalised_folder_path / type
    file_names = sorted(folder_path.glob('*.wav'))
    results = {}
    rms = [None] * len(file_names)
    distances = [None] * len(file_names)
    i = 0
    for file_name in file_names:
        distance = file_name.name[file_name.name.find('dist-') + len('dist-'):file_name.name.rfind('.wav')]
        sig = slab.Sound(file_name)
        sig.data = sig.data[0:int(44100*0.3)]
        rms[i] = numpy.average(sig.level)
        distances[i] = int(distance)
        i += 1
    rms.sort()
    distances.sort()
    plt.plot(distances, rms, '-o', label=type)
plt.title("Sound energy per distance")
plt.legend()
plt.show()