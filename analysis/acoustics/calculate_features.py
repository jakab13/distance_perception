import slab
import pathlib
import os
import pandas as pd
import re
from os.path import join
import pyloudnorm as pyln
from pydub import AudioSegment
import numpy as np

SAMPLERATE = 44100
slab.Signal.set_default_samplerate(SAMPLERATE)
DIR = pathlib.Path(__file__).parent.parent.absolute()

csv_file_name = 'loudnesses.csv'
csv_file_path = DIR / 'acoustics' / csv_file_name
COLUMN_NAMES = ["dist_group", "vocalist", "duration", "vocoding_bandwith", "RMS", "LUFS", "dbFS", "centroid", "flatness"]
vocoded_directory = DIR / 'experiment' / 'samples' / 'VEs' / 'vocoded'


def get_file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if not f.startswith('.'):
                yield pathlib.Path(join(dirpath, f))


def get_RMS(file_path, duration=None):
    sound = slab.Sound(file_path)
    duration = sound.n_samples if None else slab.Signal.in_samples(duration, sound.samplerate)
    sound.data = sound.data[:duration]
    return sound.level.mean()


def get_LUFS(file_path, duration=None):
    meter = pyln.Meter(SAMPLERATE, block_size=0.200)
    sound = slab.Binaural(file_path)
    duration = sound.n_samples if None else slab.Signal.in_samples(duration, sound.samplerate)
    sound.data = sound.data[:duration]
    return meter.integrated_loudness(sound.data)


def get_dbFS(file_path, duration=None):
    sound = AudioSegment.from_file(file_path, "wav")
    duration = sound.n_samples if None else slab.Signal.in_samples(duration, SAMPLERATE)
    sound = sound[:duration]
    return sound.dBFS


def get_distance(file_path):
    file_name = file_path.name
    distance_string = file_name[file_name.find('dist-') + len('dist-'):file_name.rfind('.wav')]
    distance = int(re.findall('\d+', distance_string)[0])
    return distance


def get_duration(file_path):
    sound = slab.Sound(file_path)
    return sound.duration


def get_speactral_feature(file_path, feature_name):
    sound = slab.Sound(file_path)
    feature = np.asarray(sound.spectral_feature(feature_name))
    feature_avg = feature.mean()
    return feature_avg


def get_vocoding_bandwidth(file_path):
    bandwidth = None
    file_name = file_path.name
    bandwidth_string = file_name[file_name.find('V-') + len('V-'):file_name.rfind('.wav')]
    if bandwidth_string:
        bandwidth = float(re.findall('\d+', bandwidth_string)[0] + "." + re.findall('\d+', bandwidth_string)[1])
    return bandwidth


def get_vocalist(file_path):
    vocalist = None
    file_name = file_path.name
    vocalist_string = file_name[file_name.find('vocalist-') + len('vocalist-'):file_name.rfind('.wav')]
    if vocalist_string:
        vocalist = int(re.findall('\d+', vocalist_string)[0])
    return vocalist


vocoded_file_paths = [f for f in get_file_paths(vocoded_directory)]
loudnesses = {f.name: {} for f in get_file_paths(vocoded_directory)}

for vocoded_file_path in vocoded_file_paths:
    loudnesses[vocoded_file_path.name] = {
        "RMS": get_RMS(vocoded_file_path),
        "LUFS": get_LUFS(vocoded_file_path),
        "dbFS": get_dbFS(vocoded_file_path),
        "dist_group": get_distance(vocoded_file_path),
        "duration": get_duration(vocoded_file_path),
        "centroid": get_speactral_feature(vocoded_file_path, "centroid"),
        "flatness": get_speactral_feature(vocoded_file_path, "flatness"),
        "vocoding_bandwith": get_vocoding_bandwidth(vocoded_file_path),
        "vocalist": get_vocalist(vocoded_file_path)
    }

df = pd.DataFrame.from_dict(loudnesses, columns=COLUMN_NAMES, orient="index")
df = df.round(decimals=5)
df.to_csv(f'analysis/acoustics/{vocoded_directory.name}_features.csv')

