import sys
path = "D:\Projects\eeg_tools\src\eeg_tools"
sys.path.append(path)
import analysis
import settings
import mne

evokeds, evokeds_avrgd = analysis.get_evokeds(settings.ids, settings.root_dir)

evokeds_avrgd["pinknoise/1"].plot_joint()
evokeds_avrgd["pinknoise/1"].plot_topo()
evokeds.pop("deviant")
mne.viz.plot_compare_evokeds(evokeds, picks="FCz")
evoked_diff = mne.combine_evoked(
    [evokeds_avrgd["pinknoise/1"], evokeds_avrgd["pinknoise/5"]], weights=[1, -1])
evoked_diff.plot_joint()
