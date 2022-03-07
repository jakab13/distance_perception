import os
import pathlib
import glob
import numpy as np
import mne
import matplotlib.pyplot as plt

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

DIR = pathlib.Path(os.getcwd())
folder_name = 'pilot/6hz79j'
folder_path = DIR / 'analysis' / 'data' / folder_name

print(folder_path)

header_files = folder_path.glob('*.vhdr')
raw_files = []

for header_file in header_files:
    raw_files.append(mne.io.read_raw_brainvision(header_file, preload=True))

raw = mne.concatenate_raws(raw_files)
events = mne.events_from_annotations(raw)
events = events[0]

raw.filter(0.5, 40)

ica = mne.preprocessing.ICA(n_components=0.99, method="fastica")
ica.fit(raw)

tmin = -0.3
tmax = 0.7
baseline = (-0.2, 0)
drops = []
reject_criteria = dict(eeg=200e-6)
flat_criteria = dict(eeg=1e-6)
event_id = {
    'deviant': 1,
    'control': 2,
    'distance/2m': 3,
    'distance/4m': 4,
    'distance/8m': 5,
    'distance/16m': 6,
    'button_press': 7
}
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, reject=reject_criteria, flat=flat_criteria,
                    reject_by_annotation=True, baseline=baseline, preload=True)

deviant = epochs['deviant'].average()
control = epochs['control'].average()
distance_all = epochs['distance'].average()
distance_2m = epochs['distance/2m'].average()
distance_4m = epochs['distance/4m'].average()
distance_8m = epochs['distance/8m'].average()
distance_16m = epochs['distance/16m'].average()

mne.viz.plot_compare_evokeds([
    # control,
    distance_2m,
    distance_4m,
    distance_8m,
    distance_16m
    ],
    legend='upper left',
    show_sensors='upper right'
)

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