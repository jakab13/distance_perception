import os
import pathlib
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt
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

events = mne.events_from_annotations(raw)
events = events[0]

raw.filter(0.5, 40)

montage = mne.channels.make_standard_montage('easycap-M1')
channel_dict = {"1": 'Fp1', "2": 'AF7', "3": 'AF3', "4": 'F1', "5": 'F3', "6": 'F5', "7": 'F7', "8": 'FT7', "9": 'FC5', "10": 'FC3', "11": 'FC1', "12": 'C1', "13": 'C3', "14": 'C5', "15": 'T7', "16": 'TP7', "17": 'CP5', "18": 'CP3', "19": 'CP1', "20": 'P1', "21": 'P3', "22": 'P5', "23": 'P7', "24": 'P9', "25": 'PO7', "26": 'PO3', "27": 'O1', "28": 'Iz', "29": 'Oz', "30": 'POz', "31": 'Pz', "32": 'CPz', "33": 'Fpz', "34": 'Fp2', "35": 'AF8', "36": 'AF4', "37": 'AFz', "38": 'Fz', "39": 'F2', "40": 'F4', "41": 'F6', "42": 'F8', "43": 'FT8', "44": 'FC6', "45": 'FC4', "46": 'FC2', "47": 'FCz', "48": 'Cz', "49": 'C2', "50": 'C4', "51": 'C6', "52": 'T8', "53": 'TP8', "54": 'CP6', "55": 'CP4', "56": 'CP2', "57": 'P2', "58": 'P4', "59": 'P6', "60": 'P8', "61": 'P10', "62": 'PO8', "63": 'PO4', "64": 'O2'}
ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7','FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
ch_types = ['eeg'] * 64
sfreq = raw.info['sfreq']
info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
info.set_montage('biosemi64')

custom_raw = mne.io.RawArray(raw.get_data(), info)

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

# fig = mne.viz.plot_events(events, event_id=event_id, sfreq=raw.info['sfreq'],
#                           first_samp=raw.first_samp)
#
# evoked_diff_2m = mne.combine_evoked([control, distance_2m], weights=[1, -1])
# evoked_diff_4m = mne.combine_evoked([control, distance_4m], weights=[1, -1])
# evoked_diff_8m = mne.combine_evoked([control, distance_8m], weights=[1, -1])
# evoked_diff_16m = mne.combine_evoked([control, distance_16m], weights=[1, -1])
#
# mne.viz.plot_compare_evokeds([
#     evoked_diff_2m,
#     evoked_diff_4m,
#     evoked_diff_8m,
#     evoked_diff_16m
# ])