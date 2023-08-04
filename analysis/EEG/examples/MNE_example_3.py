from matplotlib import pyplot as plt
import numpy as np
import mne
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])


raw = mne.io.read_raw_fif(mne.datasets.sample.data_path() +
                          '/MEG/sample/sample_audvis_raw.fif', preload=True)
events = mne.read_events(mne.datasets.sample.data_path() +
                         '/MEG/sample/sample_audvis_raw-eve.fif')
# 1 	Response to left-ear auditory stimulus
# 2 	Response to right-ear auditory stimulus
# 3 	Response to left visual field stimulus
# 4 	Response to right visual field stimulus
# 5 	Response to the smiley face
# 32    Response triggered by the button press
raw.filter(1, 40)
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False)

# Addendum to yesterdays course: A empiric approach to treshold rejection
# To make a systematic comparison, choose a signal interval (what we are
# interested in) and a noise interval (what we are not interested in). And
# compare the amount of activity in each. One way to do this is the
# (square-)Root of the Mean of the Squared sample values (RMS).

# Discuss: Why square and then take the root? Why not just take the mean?

# Lets do this for 50 different thresholds fill_between 200 and 50 microvolts
thresholds = np.linspace(200, 50, 50)
snr = np.zeros(len(thresholds))
for i, thresh in enumerate(thresholds):
    epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, baseline=None,
                        reject=dict(eeg=thresh*1e-6), proj=False, picks=picks,
                        preload=True)

    signal = epochs.copy().crop(0.0, 0.3).average()
    noise = epochs.copy().crop(None, 0.0).average()
    signal_rms = np.sqrt(np.mean(signal._data**2))
    noise_rms = np.sqrt(np.mean(noise._data**2))
    snr[i] = signal_rms/noise_rms
plt.plot(thresholds, snr)
plt.xlim(200, 50)
plt.xlabel("treshold in microvolts")
plt.ylabel("signal to noise ratio")
plt.show()

# Discuss: Why does the snr first increase then decrease?

# Now let's look at the ERP:
thresholds[snr == max(snr)][0]
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, baseline=None,
                    reject=dict(eeg=thresholds[snr == max(snr)][0] * 1e-6),
                    proj=False, picks=picks, preload=True)
epochs.average().plot()

# EEG data is recorded with a reference electrode. So the voltage values of
# each channel are actually the difference between that channel and the
# reference: Ch1 = Ch1_raw - Ref. We can re-reference the data by subtracting
# another channel (or a combination of channels) from each channel. The old
# refrence will get canceled out. Say we want to apply Ch2 as new reference:
# Ch1 - Ch2 <--> Ch1_raw-Ref - Ch2_raw-Ref <--> Ch1_raw - Ch2_raw - Ref - Ref

# To pick a reference channel let's first have a look at their positions:
epochs.plot_sensors(show_names=True)

# Let's compare a few different ones:
fig, axes = plt.subplots(2, 2)
epochs.plot_sensors(show_names=["EEG 012", "EEG 017", "EEG 024"],
                    axes=axes[0, 0])
references = [["EEG 012"], ["EEG 017", "EEG 024"], "average"]
for ref, ax in zip(references, [axes[0, 1], axes[1, 0], axes[1, 1]]):
    epochs.set_eeg_reference(ref)
    epochs.average().plot(axes=ax, show=False)
    ax.set_title("reference electrode: %s" % (ref))

# Discuss: Which channel is the appropriate reference?


# Usually we are interested in the difference between experimental conditons.
# lets compare the response to the same sound played to the left and right ear
evoked_left = epochs["1"].average()
evoked_right = epochs["2"].average()
fig, ax = plt.subplots(2)
evoked_left.plot(axes=ax[0], show=False)
evoked_right.plot(axes=ax[1], show=False)
plt.show()

# Discuss: What might be an interesting effect that you could investigate?


# Compare activity between left and right "hemisphere"

lh_channels = ["EEG 001", "EEG 004", "EEG 005", "EEG 008", "EEG 009",
               "EEG 010", "EEG 011", "EEG 017", "EEG 018", "EEG 019",
               "EEG 020", "EEG 025", "EEG 026", "EEG 027", "EEG 028",
               "EEG 029", "EEG 036", "EEG 037", "EEG 038", "EEG 039",
               "EEG 044", "EEG 045", "EEG 046", "EEG 047",
               "EEG 054", "EEG 057"]

rh_channels = ["EEG 003", "EEG 006", "EEG 007", "EEG 013", "EEG 014",
               "EEG 015", "EEG 016", "EEG 021", "EEG 022", "EEG 023",
               "EEG 024", "EEG 031", "EEG 032", "EEG 033", "EEG 034",
               "EEG 035", "EEG 040", "EEG 041", "EEG 042", "EEG 043",
               "EEG 049", "EEG 050", "EEG 051", "EEG 052",
               "EEG 055", "EEG 056"]

evoked_left_lh = evoked_left.copy().pick_channels(lh_channels).crop(0.0, 0.3)
evoked_left_rh = evoked_left.copy().pick_channels(rh_channels).crop(0.0, 0.3)
evoked_right_lh = evoked_right.copy().pick_channels(lh_channels).crop(0.0, 0.3)
evoked_right_rh = evoked_right.copy().pick_channels(rh_channels).crop(0.0, 0.3)


rms = [np.sqrt(np.mean(evoked_left_lh.data**2)),
       np.sqrt(np.mean(evoked_right_lh.data**2)),
       np.sqrt(np.mean(evoked_left_rh.data**2)),
       np.sqrt(np.mean(evoked_right_rh.data**2))]

x = ["left_lh", "right_lh", "left_rh", "right_rh"]
plt.bar(x, rms)
plt.show()
