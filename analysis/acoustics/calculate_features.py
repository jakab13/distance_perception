import matplotlib.pyplot as plt
import slab
import pathlib
import os
import pandas as pd
import re
from os.path import join
import pyloudnorm as pyln
from pydub import AudioSegment
import numpy as np
from librosa.feature import spectral_centroid

SAMPLERATE = 44100
slab.Signal.set_default_samplerate(SAMPLERATE)
DIR = pathlib.Path(os.getcwd())

csv_file_name = 'features.csv'
csv_file_path = DIR / 'acoustics' / csv_file_name
COLUMN_NAMES = [
    "dist_group",
    "vocalist",
    "duration",
    "vocoding_bandwith",
    "RMS",
    "LUFS",
    "dbFS",
    "centroid",
    "flatness"
]

USO_COLUMN_NAMES = [
    "USO_id",
    "RMS",
    "LUFS",
    "dbFS",
    "dist_group",
    "centroid",
    "flatness",
    "onset_slope",
    "time_cog",
    "spectral_slope",
    "centroid_control",
    "onset_delay"
]

vocoded_directory = DIR / 'experiment' / 'samples' / 'VEs' / 'vocoded'
USO_directory = DIR / 'experiment' / 'samples' / 'USOs' / 'final_USOs'
USO_distance_ranges = {
    1: [18, 19, 20, 21, 22],
    2: [162, 171, 180, 189, 198],
    3: [450, 475, 500, 525, 550],
    4: [1080, 1140, 1200, 1260, 1320],
    5: [2250, 2375, 2500, 2625, 2750]
}


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


def get_from_file_name(file_path, pre_string):
    sub_val = None
    file_name = file_path.name
    sub_string = file_name[file_name.find(pre_string) + len(pre_string):file_name.rfind('.wav')]
    if sub_string:
        sub_val = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", sub_string)[0]
        sub_val = int(sub_val) if sub_val.isdigit() else float(sub_val)
    return sub_val


def get_duration(file_path):
    sound = slab.Sound(file_path)
    return sound.duration


def get_control_file_path(file_path):
    file_name = file_path.name
    if "control" not in file_name:
        sub_string = file_name[:file_name.find("_dist")]
        control_file_name = sub_string + "_control.wav"
        control_file_path = file_path.parent / control_file_name
    else:
        control_file_path = file_path
    return control_file_path


def get_spectral_feature(file_path, feature_name, mean="rms", duration=None, control=False):
    file_path = file_path if not control else get_control_file_path(file_path)
    sound = slab.Sound(file_path)
    if duration is not None:
        duration = slab.Signal.in_samples(duration, sound.samplerate)
        sound.data = sound.data[:duration]
    feature = np.asarray(sound.spectral_feature(feature_name, mean=mean))
    feature_avg = feature.mean()
    return feature_avg


def get_spectral_slope(file_path, duration=None):
    sound = slab.Sound(file_path)
    if duration is not None:
        duration = slab.Signal.in_samples(duration, sound.samplerate)
        sound.data = sound.data[:duration]
    win_length = slab.Signal.in_samples(0.3, sound.samplerate)
    centroids = spectral_centroid(y=sound.data.T, sr=sound.samplerate, n_fft=win_length)
    centroids = centroids.transpose(2, 0, 1)
    centroids = centroids.squeeze()
    slope = np.gradient(centroids, axis=0).mean()
    return slope


def get_USO_distance_group(file_path):
    group = None
    distance = get_from_file_name(file_path, "dist-")
    for d_group, d_range in USO_distance_ranges.items():
        if distance in d_range:
            group = d_group
    return group


def get_onset_slope(file_path):
    sound = slab.Sound(file_path)
    return sound.onset_slope()


def get_time_cog(file_path, duration=1.0, resamplerate=48828):
    sound = slab.Sound(file_path)
    if sound.samplerate != resamplerate:
        sound = sound.resample(resamplerate)
    duration = slab.Signal.in_samples(duration, sound.samplerate)
    sound.data = sound.data[:duration]
    env = sound.envelope()
    cog_l = np.average(np.arange(0, len(env[:, 0])), weights=env[:, 0])
    cog_r = np.average(np.arange(0, len(env[:, 1])), weights=env[:, 1])
    cog = np.average([cog_l, cog_r])
    cog /= sound.samplerate
    return cog


def onset_delay(file_path):
    sound = slab.Sound(file_path)
    delay = next((i for i, x in enumerate(sound.data[:, 0]) if x), None)
    delay_in_s = delay / sound.samplerate
    return delay_in_s


vocoded_file_paths = [f for f in get_file_paths(vocoded_directory)]
features = {f.name: {} for f in get_file_paths(vocoded_directory)}

USO_file_paths = [f for f in get_file_paths(USO_directory)]
USO_features = {f.name: {USO_key: None for USO_key in USO_COLUMN_NAMES} for f in get_file_paths(USO_directory)}

for USO_file_path in USO_file_paths:
    USO_features[USO_file_path.name]["USO_id"] = get_from_file_name(USO_file_path, "ms_")
    USO_features[USO_file_path.name]["RMS"] = get_RMS(USO_file_path)
    USO_features[USO_file_path.name]["LUFS"] = get_LUFS(USO_file_path)
    USO_features[USO_file_path.name]["dbFS"] = get_dbFS(USO_file_path)
    USO_features[USO_file_path.name]["dist_group"] = get_USO_distance_group(USO_file_path)
    USO_features[USO_file_path.name]["centroid"] = get_spectral_feature(USO_file_path, "centroid", duration=1.0)
    USO_features[USO_file_path.name]["flatness"] = get_spectral_feature(USO_file_path, "flatness", duration=1.0)
    USO_features[USO_file_path.name]["onset_slope"] = get_onset_slope(USO_file_path)
    USO_features[USO_file_path.name]["spectral_slope"] = get_spectral_slope(USO_file_path)
    USO_features[USO_file_path.name]["time_cog"] = get_time_cog(USO_file_path)
    USO_features[USO_file_path.name]["centroid_control"] = get_spectral_feature(USO_file_path, "centroid", control=True)
    USO_features[USO_file_path.name]["onset_delay"] = onset_delay(USO_file_path)

for vocoded_file_path in vocoded_file_paths:
    features[vocoded_file_path.name] = {
        "RMS": get_RMS(vocoded_file_path),
        "LUFS": get_LUFS(vocoded_file_path),
        "dbFS": get_dbFS(vocoded_file_path),
        "dist_group": get_from_file_name(vocoded_file_path, "dist-"),
        "duration": get_duration(vocoded_file_path),
        "centroid": get_spectral_feature(vocoded_file_path, "centroid"),
        "flatness": get_spectral_feature(vocoded_file_path, "flatness"),
        "vocoding_bandwith": get_from_file_name(vocoded_file_path, "V-"),
        "vocalist": get_from_file_name(vocoded_file_path, "vocalist-")
    }



df = pd.DataFrame.from_dict(features, columns=COLUMN_NAMES, orient="index")
df = df.round(decimals=5)
df.to_csv(f'analysis/acoustics/{vocoded_directory.name}_features.csv')

df = pd.DataFrame.from_dict(USO_features, columns=USO_COLUMN_NAMES, orient="index")
df = df.round(decimals=5)
df.to_csv('analysis/acoustics/USO_features_2.csv')

df = pd.read_csv(DIR / 'analysis' / 'acoustics' / 'USO_features.csv')
features_dict = df.to_dict()

excludes = [
    {"USO_id": 3, "condition": 4},
    {"USO_id": 3, "condition": 5},
    {"USO_id": 5, "condition": 5},
    {"USO_id": 10, "condition": 2},
    {"USO_id": 10, "condition": 3},
    {"USO_id": 14, "condition": 5},
    {"USO_id": 17, "condition": 1},
    {"USO_id": 17, "condition": 3},
    {"USO_id": 17, "condition": 4},
    {"USO_id": 19, "condition": 3},
    {"USO_id": 20, "condition": 4},
    {"USO_id": 25, "condition": 2},
    {"USO_id": 25, "condition": 3},
    {"USO_id": 25, "condition": 4},
    {"USO_id": 25, "condition": 5}
]

onset_delay_avg = {}
for USO_idx, USO_id in features_dict["USO_id"].items():
    dist_group = features_dict["dist_group"][USO_idx]
    if dist_group == dist_group:
        is_excluded = False
        for exclude in excludes:
            if exclude["USO_id"] == USO_id and exclude["condition"] == dist_group:
                is_excluded = True
                onset_delay_avg[USO_idx] = None
        if not is_excluded:
            dist_group_idxs = [k for k, v in features_dict["dist_group"].items() if v == dist_group]
            group_idxs = [k for k, v in features_dict["USO_id"].items() if v == USO_id]
            final_idxs = list(set(dist_group_idxs).intersection(group_idxs))
            onset_delays = []
            for idx in final_idxs:
                onset_delays.append(features_dict["onset_delay"][idx])
            onset_delay_avg[USO_idx] = np.asarray(onset_delays).mean()
        else:
            print("excluded", USO_idx)
    else:
        onset_delay_avg[USO_idx] = None

df["onset_delay_avg"] = onset_delay_avg

