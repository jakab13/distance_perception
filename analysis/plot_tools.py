import matplotlib.pyplot as plt
import pathlib
import os
from os.path import join
import numpy as np
import slab
import re

DIR = pathlib.Path(os.getcwd())
USO_directory = DIR / 'experiment' / 'samples' / 'USOs' / 'final_USOs'

SAMPLERATE = 44100
slab.Signal.set_default_samplerate(SAMPLERATE)
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


def get_envelope(file_path):
    sound = slab.Sound(file_path)
    env = sound.envelope()
    return env


def avg_envs(file_paths, mono=False):
    avg_env = None
    if type(file_paths) != list:
        file_paths = [file_paths]
    for file_path in file_paths:
        env = get_envelope(file_path)
        if avg_env is None:
            avg_env = env
        else:
            avg_env += env
    avg_env /= len(file_paths)
    if mono:
        avg_env.data = avg_env.data.mean(axis=1)
    return avg_env


def get_from_file_name(file_path, pre_string):
    sub_val = None
    file_name = file_path.name
    sub_string = file_name[file_name.find(pre_string) + len(pre_string):file_name.rfind('.wav')]
    if sub_string:
        sub_val = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", sub_string)[0]
        sub_val = int(sub_val) if sub_val.isdigit() else float(sub_val)
    return sub_val


def get_USO_distance_group(file_path):
    group = None
    distance = get_from_file_name(file_path, "dist-")
    for d_group, d_range in USO_distance_ranges.items():
        if distance in d_range:
            group = d_group
    return group


def filter_paths(file_paths, condition, USO_id):
    out = []
    for file_path in file_paths:
        dist_group = get_USO_distance_group(file_path)
        id = get_from_file_name(file_path, "ms_")
        if dist_group == condition:
            if id in USO_id:
                out.append(file_path)
    return out


USO_file_paths = [f for f in get_file_paths(USO_directory) if 'control' not in f.name]

fails = [7, 10, 11, 14, 17, 18, 22, 28]
correct = [2, 3, 5, 19, 20, 21, 25, 27]
all_USO_ids = [7, 10, 11, 14, 17, 18, 22, 28, 2, 3, 5, 19, 20, 21, 25, 27]

file_paths_conditions = [filter_paths(USO_file_paths, condition=i, USO_id=all_USO_ids) for i in [1, 2, 3, 4, 5]]
condition_1s = filter_paths(USO_file_paths, condition=1, USO_id=all_USO_ids)
condition_2s = filter_paths(USO_file_paths, condition=2, USO_id=all_USO_ids)
condition_3s = filter_paths(USO_file_paths, condition=3, USO_id=all_USO_ids)
condition_4s = filter_paths(USO_file_paths, condition=4, USO_id=all_USO_ids)
condition_5s = filter_paths(USO_file_paths, condition=5, USO_id=all_USO_ids)
envs = [avg_envs(file_paths, mono=True) for file_paths in condition_1s]

condition_envs = dict()
for file_path in condition_5s:
    id = get_from_file_name(file_path, "ms_")
    if id not in condition_envs:
        condition_envs[id] = [get_envelope(file_path)]
    else:
        condition_envs[id].append(get_envelope(file_path))

condition_envs_avrg = dict()
for USO_id, env_list in condition_envs.items():
    condition_envs_avrg[USO_id] = avg_envs(env_list, mono=True)

for USO_id, envs in condition_envs.items():
    # if USO_id in fails:
    plt.plot(envs[0].data[:, 0], label=f"USO_id {USO_id}")
plt.xlim([0, SAMPLERATE * 1.0])
plt.xticks(np.linspace(0, 44100, 5), np.linspace(0, 1, 5))
plt.xlabel("Time (s)")
plt.ylabel("RMS")
plt.legend()
plt.title("Average USO sound energy - Condition 5")
plt.show()

for USO_id in all_USO_ids:
    for real_distance in condition_envs[USO_id]:
        plt.plot(real_distance.data[:, 0])
    plt.xlim([0, SAMPLERATE * 1.0])
    plt.xticks(np.linspace(0, 44100, 5), np.linspace(0, 1, 5))
    plt.xlabel("Time (s)")
    plt.ylabel("RMS")
    plt.title(f"Average USO sound energy - Condition 5 - USO {USO_id}")
    plt.savefig(f"Average USO sound energy - Condition 5 - USO  {USO_id}")
    plt.clf()
    # plt.show()


for idx, env in enumerate(envs):
    plt.plot(env, label=f"Condition {idx+1}")
plt.xlim([0, SAMPLERATE * 1.0])
plt.xticks(np.linspace(0, 44100, 5), np.linspace(0, 1, 5))
plt.xlabel("Time (s)")
plt.ylabel("RMS")
plt.legend()
plt.title("Average USO sound energy per distance")
plt.show()

curr_uso_idx = 16
for USO_file_path in USO_file_paths:
    dist_group = get_USO_distance_group(USO_file_path)
    id = get_from_file_name(USO_file_path, "ms_")
    if id == all_USO_ids[curr_uso_idx]:
        idx = all_USO_ids.index(id)
        env = get_envelope(USO_file_path)
        env.data = env.data.mean(axis=1)
        plt.figure((idx + 1) * dist_group)
        plt.plot(env.data)
        plt.xlim([0, SAMPLERATE * 1.0])
        plt.xticks(np.linspace(0, 44100, 5), np.linspace(0, 1, 5))
        plt.xlabel("Time (s)")
        plt.ylabel("RMS")
        plt.title(f"USO {id} - Condition {dist_group}")
        plt.savefig(f"analysis/acoustics/figures/USO {id} - Condition {dist_group}")
plt.show()
