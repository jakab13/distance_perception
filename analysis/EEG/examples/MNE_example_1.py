#### EEG data processing with MNE Python - example script (EEG Practical Neuro 2, SS2020) ####
# Authors: Ole Bialas & Tilman Stephani

# For a guide on installing python and mne see:
# https://mne.tools/stable/install/mne_python.html
# For a detailed description of MNE-python you can read the paper
# Gramfort et al. (2014): MNE software for processing MEG and EEG data or
# see their website: https://mne.tools/0.13/tutorials.html

# Import needed modules
import mne

# calling data_path will automatically download the data:
data_path = mne.datasets.sample.data_path(verbose=True)

# This is the sample data set provided by MNE. MEG data were acquired
# with a Neuromag VectorView system (Elekta Oy, Helsinki, Finland) with
# 306 sensors arranged in 102 triplets, each comprising two orthogonal planar
# gradiometers and one magnetometer. EEG data were recorded simultaneously using an
# MEG-compatible cap with 60 electrodes. In the experiment, auditory stimuli
# (delivered monaurally to the left or right ear) and visual stimuli
# (shown in the left or right visual hemifield) were presented in a random
# sequence with a stimulus-onset asynchrony (SOA) of 750 ms. To control for the
# subject's attention, a smiley face was presented intermittently and the subject
# was asked to press a button upon its appearance.
# In this tutorial we are only using the data from the EEG channels. However,
# everything shown here can be applied to the MEG data as well so feel free to try
# it out :)

# load raw data
raw = mne.io.read_raw_fif(data_path+'/MEG/sample/sample_audvis_raw.fif', preload=True)

# The MNE data structure does not only contain the sensor time series data but
# also a lot of meta information.

raw.info
raw.info["chs"]  # for example a list of all channels
raw.info["dig"]  # or coordinates of points on the surface of the subjects head

# The dataset contains MEG and EEG data. For now we are only intereted in the
# latter so we will select it:
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)
raw.pick(picks=picks)

# plot  and inspect raw data. Are there any data segments that
# should be removed for further analyis?
raw.plot()

# Load the events. This is a table that contains the time points at which a
# stimulus was played and it's type. The stimulus type is encoded by a number:
# 1 	Response to left-ear auditory stimulus
# 2 	Response to right-ear auditory stimulus
# 3 	Response to left visual field stimulus
# 4 	Response to right visual field stimulus
# 5 	Response to the smiley face
# 32 Response triggered by the button press
events = mne.read_events(data_path+'/MEG/sample/sample_audvis_raw-eve.fif')

# To remove power line noise, we apply a lowpass filter with a 40 Hz cutoff
raw.filter(None, 40)
help(raw.filter)  # get info on what the function does

# WARNING: filtering alters the data and can produice artifacts and lead
# to incorrect conclusions. It is thus important to understand the effect of
# filters applied. We will ignore this for now but you can see this paper:
# Widmann et al. (2015): Digital filter design for electrophysiological data – a practical approach.

raw.plot()  # plot data again, can you see the difference?

# We will now cut the data into equally long segments arround the stimuli to
# obtain the so-called epochs. For this we need to define some parameters:
tmin = -0.7  # start of the epoch (relative to the stimulus)
tmax = 0.7  # end of the epoch
event_id = dict(vis_l=3, vis_r=2)  # the stimuli we are interested in
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=True)  # get epochs

# We can now average over the epochs to obtain the ERP (= event-related potential).
# This is what we will work with in the next section.
epochs.average().plot()

# However, this does not look right. There are big differences in the channels' baseline
# which overshadows the actual ERP. We can circumvent this by setting a channel-
# specific baseline. To do this we select a certain interval in the Epoch. For
# each channel, the mean of that interval will be subtracted from the rest of
# the epoch.
baseline = (-0.2, 0)
epochs.apply_baseline(baseline)
epochs.average().plot()
# The baseline can also be given as an argument when computing the epochs but
# we did not do this here for educational reasons.

# Now you can play with the parameters (start, stop and baseline) and see how
# the evoked response changes. For example: Try choosing an interval that
# contains more than one stimulus. Try using an interval with strong stimulus
# related activity as baseline.

# Assignment 1: plot the epoch average with different filter configurations,
# epoch intervals (tmin / tmax) and baseline intervals
