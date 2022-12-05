import os
import pathlib
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

DIR = pathlib.Path(os.getcwd())
project_name = 'pilot'
with open(DIR / 'analysis' / 'data' / project_name / 'participants.json', 'r') as file:
    participants_dict = json.load(file)
participants_ids = participants_dict['participant_ids']

participants_id = participants_ids[0]

folder_path = DIR / 'analysis' / 'data' / project_name / participants_id

header_files = folder_path.glob('*.vhdr')
raw_files = []

for header_file in header_files:
    raw_files.append(mne.io.read_raw_brainvision(header_file, preload=True))

raw = mne.concatenate_raws(raw_files)

# TODO: Bad channels should be identified and saved right after the 'raw' dictionary has been created
# It is also recommended to use the Autoreject toolbox for bad channel selection and interpolation

ch_dict = {"1": 'Fp1', "2": 'Fp2', "3": 'F7', "4": 'F3', "5": 'Fz', "6": 'F4', "7": 'F8', "8": 'FC5',
           "9": 'FC1', "10": 'FC2', "11": 'FC6', "12": 'T7', "13": 'C3', "14": 'Cz', "15": 'C4', "16": 'T8',
           "17": 'TP9', "18": 'CP5', "19": 'CP1', "20": 'CP2', "21": 'CP6', "22": 'TP10', "23": 'P7', "24": 'P3',
           "25": 'Pz', "26": 'P4', "27": 'P8', "28": 'PO9', "29": 'O1', "30": 'Oz', "31": 'O2', "32": 'PO10',
           "33": 'AF7', "34": 'AF3', "35": 'AF4', "36": 'AF8', "37": 'F5', "38": 'F1', "39": 'F2', "40": 'F6',
           "41": 'FT9', "42": 'FT7', "43": 'FC3', "44": 'FC4', "45": 'FT8', "46": 'FT10', "47": 'C5', "48": 'C1',
           "49": 'C2', "50": 'C6', "51": 'TP7', "52": 'CP3', "53": 'CPz', "54": 'CP4', "55": 'TP8', "56": 'P5',
           "57": 'P1', "58": 'P2', "59": 'P6', "60": 'PO7', "61": 'PO3', "62": 'POz', "63": 'PO4', "64": 'PO8'}
ch_names = [ch_name for ch_number, ch_name in ch_dict.items()]
ch_types = ['eeg'] * 64
sfreq = raw.info['sfreq']
info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

# Montage #1 using the 'biosemi64' setup which has 64 channels by default
info.set_montage('biosemi64')

# Montage #2 using the 'easycap-M1' setup
# based on https://github.com/fieldtrip/fieldtrip/blob/master/template/layout/acticap-64ch-standard2.mat
# montage = mne.channels.make_standard_montage('easycap-M1')

# Montage #3 using the 'standard_1020' setup and keeping only the relevant channels
info.set_montage('standard_1020')

custom_raw = mne.io.RawArray(raw.get_data(), info)
custom_montage = custom_raw.get_montage()

fig = custom_montage.plot(kind='3d')
fig.gca().view_init(azim=70, elev=15)
custom_montage.plot(kind='topomap')

events = mne.events_from_annotations(raw)
events = events[0]

raw.filter(0.5, 40)

ica = mne.preprocessing.ICA(n_components=0.99, method="fastica")
ica.fit(custom_raw)

tmin = -0.15
tmax = 1.35
baseline = (-0.15, 0)
drops = []
reject_criteria = dict(eeg=200e-6)
flat_criteria = dict(eeg=1e-6)
event_id = {
    'deviant': 1,
    'control': 2,
    'distance/20': 3,
    'distance/200': 4,
    'distance/1000': 5,
    'distance/2000': 6,
    'button_press': 7
}

epochs = mne.Epochs(custom_raw, events, event_id, tmin, tmax, reject=reject_criteria, flat=flat_criteria,
                    reject_by_annotation=True, baseline=baseline, preload=True)

deviant = epochs['deviant'].average()
control = epochs['control'].average()
distance = epochs['distance'].average()
distance_20 = epochs['distance/20'].average()
distance_200 = epochs['distance/200'].average()
distance_1000 = epochs['distance/1000'].average()
distance_2000 = epochs['distance/2000'].average()

fig, ax = plt.subplots(5)
plt.title(participants_id)
control.plot(axes=ax[0], show=False)
distance_20.plot(axes=ax[1], show=False)
distance_200.plot(axes=ax[2], show=False)
distance_1000.plot(axes=ax[3], show=False)
distance_2000.plot(axes=ax[4], show=False)
plt.show()

time_unit = dict(time_unit="s")
# evoked.plot_joint(title="Compare evokeds", ts_args=time_unit, topomap_args=time_unit)
adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, "eeg")
plt.matshow(adjacency.toarray())

evoked = mne.combine_evoked([control, distance_2000], weights=[-1, 1])
X = [epochs['distance/20'].get_data().transpose(0, 2, 1),
     epochs['distance/1000'].get_data().transpose(0, 2, 1)]
t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(
    X, threshold=dict(start=.2, step=.2), adjacency=adjacency, n_permutations=100)
significant_points = cluster_pv.reshape(t_obs.shape).T < .05
evoked.plot_image(mask=significant_points, show_names="all", titles="Distance_20 vs Distance_1000", **time_unit)

# mne.viz.plot_compare_evokeds([
#     control,
#     # distance,
#     distance_20,
#     distance_200,
#     distance_1000,
#     distance_2000
#     ],
#     title=participants_id
# )


# mne.viz.plot_compare_evokeds([
#     control,
#     distance_all
#     ],
#     legend='upper left',
#     show_sensors='upper right'
# )
