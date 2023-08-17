#### EEG data processing with MNE Python - example script (EEG Practical Neuro 2, SS2020) ####
# Authors: Ole Bialas & Tilman Stephani

# DAY 2: rejection of bad channels and epochs; ICA

# For a guide on installing python and mne see:
# https://mne.tools/stable/install/mne_python.html
# For a detailed description of MNE-python you can read the paper
# Gramfort et al. (2014): MNE software for processing MEG and EEG data or
# see their website: https://mne.tools/0.13/tutorials.html


# Import needed modules
import mne

# get the example data:
data_path = mne.datasets.sample.data_path(verbose=True)
raw = mne.io.read_raw_fif(data_path+'/MEG/sample/sample_audvis_raw.fif', preload=True)
events = mne.read_events(data_path+'/MEG/sample/sample_audvis_raw-eve.fif')


## Let´s do all the previous preprocessing steps (as in MNE_example_1.py)
# select EEG data
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)
raw.pick(picks=picks)

# filter
raw.filter(None, 40)

# segmentation into epochs
tmin = -0.7  # start of the epoch (relative to the stimulus)
tmax = 0.7  # end of the epoch
event_id = dict(vis_l=3, vis_r=2)  # the stimuli we are interested in
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=True)  # get epochs

# baseline correction
baseline = (-0.2, 0)
epochs.apply_baseline(baseline)



# ...let us now continue with further preprocessing of the data. 


## Rejection of bad channels and segments

# Ideally, our ERPs would contain only brain activity. However, EEG and MEG
# sensors are subject to electromagnetic perturbances from all kinds of sources
# which can cause artifacts that deteriorate the signal quality. In cases where
# those artifacts are time-locked to the stimulus they might even completely
# change the ERP. In the following we will deal with two sources, blinks
# and non-physiological artifacts.

# Non-physiological artifacts are created by e.g. electromagnetic interference
# or mechanical displacement of channels/cables.

# On the one hand, an entire channel can be too noisy for analysis. In this case,
# one should remove it.

raw.plot() # look at the continuous EEG again and identify channels that either show consistently
# higher frequencies and higher amplitudes than other channels or no signal at all.

raw.info['bads'] += [] # add names of channels here to mark them as "bad"; e.g. 'EEG 001'
# Alternatively, you can also click on the channel names in the plot to mark them as "bad".
# For this data here, it seems we don´t need to remove a channel.

# Alternatively, you can also check for bad channels in the epoched data and indicate them there:
epochs.average().plot()
epochs.info['bads'] += []

# A possibility to "repair" a bad channel is to interpolate its signal based on the information
# from the other channels. This can be done with this command:
# raw_interpol = raw.copy().interpolate_bads() # but we don´t need it here.


# On the other hand, there might be specific epochs that we should exclude (for example, due to
# extensive movement artifacts). This we can do manually when inspecting the data:
epochs.plot() # click on the epochs to mark them as "bad"

# ...or automatically by selecting certain signal amplitude threshold criteria for the different types of data:
reject_criteria = dict(eeg=200e-6,       # 200 µV
                       eog=300e-6)       # 300 µV
flat_criteria = dict(eeg=1e-6)           # 1 µV
# Note that these values are very liberal here.

epochs_auto = mne.Epochs(raw, events, event_id, tmin=-0.7, tmax=0.7,
                    reject=reject_criteria, flat=flat_criteria,
                    reject_by_annotation=False, preload=True) # this is the same command for extracting epochs as used above
epochs_auto.plot_drop_log() # summary of rejected epochs per channel

# --> Task: Play with the parameters indicated by reject_criteria and flat_criteria. Does the number
# of excluded epochs change?



# Another completely automatized way of dealing with these artifacts is
# the AutoReject pipeline: The data is split into a train and test set.
# The test set is used to define channel specific rejection thresholds.
# Then test and train set are compared after these thresholds were applied, and
# epochs where channels have a peak-to-peak amplitude greater than that
# threshold are either repaired or rejected. The aim of the pipeline
# is to minimize the difference between test and train data. For a detailed
# description see: Jas et al. (2017): Autoreject: Automated artifact rejection
# for MEG and EEG data.
# This module can be installed as described here: http://autoreject.github.io/
# What is nice about this pipeline is that most of the parameters are set
# automatically. We can define the number of channels that are allowed to be
# interpolated (repaired). Setting the random_state makes the process
# deterministic so everyone gets the same result.

from autoreject import AutoReject # import the module
ar=AutoReject(n_interpolate=[3,6,12], random_state=42)
epochs_ar, reject_log = ar.fit_transform(epochs, return_log=True)

# Lets have a look at what AutoReject did to the data:
reject_log.plot_epochs(epochs) # one should carefully check that not too much of the data was removed...


# However for now, let´s continue working with the not-fully automatized rejection approach
# where we set the threshold manually:
epochs = epochs_auto



## Rejection of signals of no interest using Independent Component Analysis (ICA)

# Another prominent source of noise in EEG data are blink artifacts:
# The muscles that cause our eyelids to close and open generate
# electrical dipoles that are picked up by the EEG, especially by the frontal electrodes.
# A good way to detect those artifacts is Independent Component Analysis (ICA).
# This is a procedure that finds statistically independent components in the data.
# Since The blink artifacts are generated outside of the brain, they can be
# detected by ICA very reliably.

# Blink artifacts are visible in the data as sharp peaks of high amplitude, most
# prominent in frontal EEG channels.
# --> Task: Can you detect some in the EEG data? (Use either "raw.plot()" or "epochs.plot()")


# Now, let us do the independent component analysis. The number of components will
# be selected so that the cumulative explained variance is < 0.99
ica = mne.preprocessing.ICA(n_components=0.99, method="fastica")
ica.fit(epochs)

# We can project the weights from the unmixing matrix on the scalp (EEG sensors)
# to get something that looks similar to the previously computed voltage
# distributions (topographies). The magnitude at each channel tells you how much the component
# affects that channel - the sign is random.
ica.plot_components()

# The first component (they are ordered by explained variance) looks like a blink artifact.
# To be sure we can also look at the time series data. This looks like the raw
# data - only with independent sources instead of channels.
ica_sources = ica.get_sources(epochs)
ica_sources.plot(picks="all")

# The time series of the first component looks like the one from the EOG
# channel. Now that we have identified it as an artifact we can remove the component
# from our data.
epochs_ica = ica.apply(epochs, exclude=[0]) # insert the index of the bad IC here

# Now the data should not contain any blink artifacts anymore. You can control this here:
epochs_ica.plot()

# In the following, we will use the ICA-corrected data:
epochs = epochs_ica

# Let´s also save the data at this point:
epochs_fname = 'D:\Lehre\EEG_practical_Neuro2\datasample_audvis_epo.fif' # enter file path and name here
epochs.save(epochs_fname, overwrite = True)


# loading the data again can be done with this command:
epochs = mne.read_epochs(epochs_fname, preload=True)



