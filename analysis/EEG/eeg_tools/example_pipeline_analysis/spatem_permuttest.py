import mne
from mne.stats import spatio_temporal_cluster_test
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
# %matplotlib qt

# cluster-based spatio-temporal permutation test.
# All subjects:
evokeds, evokeds_avrgd = analysis.get_evokeds(settings.ids, settings.root_dir)

stats.spatem_test(evokeds, conditions=["pinknoise/1", "pinknoise/5"]
