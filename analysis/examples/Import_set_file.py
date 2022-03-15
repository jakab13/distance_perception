#### EEG data processing with MNE Python (EEG Practical Neuro 2, SS2020) ####
# Import of .set files and first exploration of the dataset from OpenNeuro

import mne
import numpy as np

## Load the data
data_path = ''  # indicate the data path here
subj_id = 'sub-001'
raw = mne.io.read_raw_eeglab(data_path + '/' + subj_id + '_task-MMN_eeg.set', preload=True)
# Note: Every .set file is accompanied by a .fdt file that contains the actual data. In order to correctly
# load the data the name of the .fdt file and the name for it indicated in the .set file must match.


# save in MNE Python format (.fif) for later analyses
data_fname = data_path + 'mydata_raw.fif'  # enter subject-specific path and name here
raw.save(data_fname, overwrite=True)

## Have a first look at the available data
# continuous raw data:
raw.plot()

# Set montage (standard montage)
montage = mne.channels.make_standard_montage('standard_1020')
raw.rename_channels({'FP1':'Fp1', 'FP2':'Fp2'}) # rename a few channels so that electrode position names match
raw.set_channel_types({'HEOG_left':'eog', 'HEOG_right':'eog', 'VEOG_lower':'eog'})
raw.set_montage(montage)


# electrode layout
raw.plot_sensors(ch_type='eeg', show_names=True) # you can also try to add the argument: kind='3d'


## Get the events present in the raw data
# in datasets originally from EEGLAB (.set files), the event information is stored here:
raw.annotations
mne.Annotations? # open the documentation to see what this variable is about

raw.annotations[3]  # look at one instance: we have 3 pieces of information: onset, duration, and description
len(raw.annotations)  # number of all events

# convert to an "Events array" (counted in sampling points)
events, event_id = mne.events_from_annotations(raw)  # all events
events, event_id = mne.events_from_annotations(raw, {'event_code_in_data1': 1, 'event_code_in_data2': 2})  # select events of interest; EVENT CODES MUST BE ADJUSTED HERE!
# --> Question: What do the event codes represent (check the information on the repository website)?

# assign more meaningful names to the event codes
event_id = dict(my_event1=1, my_event2=2) # please come up with names you find intuitive

# Inspect when which event occured
fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp, event_id=event_id)
fig.subplots_adjust(right=0.7)  # make room for legend


## Check the inter-stimulus intervals (ISI)
# calculate the ISIs from the event latencies
ISI_pt = np.diff(events[:, 0])
ISI_ms = ISI_Aud_pt / raw.info['sfreq'] * 1000  # convert data sampling points into milliseconds

# plot the distribution of ISIs
import matplotlib.pyplot as plt

plt.figure()
plt.hist(ISI_ms, bins=50)

# plot the time course of ISIs and explore it (e.g. by zooming in)
plt.figure()
plt.plot(ISI_ms)


## --> Tasks/ Questions:
# Examine the time course of the events and their ISIs: What do these plots tell us about the experimental design?


## --> Bonus task:
# Also load the data of participant 2 and inspect the event structure there.
# Does the design differ from participant 1 (probability of stimuli, ISIs, etc.)?
