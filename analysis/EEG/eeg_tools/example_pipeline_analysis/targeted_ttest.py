import mne
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt
import sys
sys.path.append("D:/Projects/eeg_tools/src/eeg_tools")
import setup_eeg_tools as set
import analysis
import settings
import stats


evokeds = analysis.get_evokeds(settings.ids, settings.root_dir, return_average=False)

# Target ttest:
time_windows = [(0.15, 0.35), (0.6, 0.8)]
electrodes = ["Fz", "FCz", "Cz", "Pz", "POz", "Oz"]
conditions=["pinknoise/1", "pinknoise/2"]

stats.target_test(evokeds, time_windows, electrodes, conditions=["pinknoise/1", "pinknoise/2"])

index = "time"
report = "{electrode}, time: {tmin}-{tmax} s; stat={statistic:.3f}, p={p}"
print("Targeted t-test results:")
for tmin, tmax in time_windows:
    cond0 = mne.grand_average(evokeds[conditions[0]]).copy().crop(tmin, tmax).to_data_frame(index=index)
    cond1 = mne.grand_average(evokeds[conditions[1]]).copy().crop(tmin, tmax).to_data_frame(index=index)
    for electrode in electrodes:
        # extract data
        A = cond0[electrode]
        B = cond1[electrode]
