#### EEG data processing with MNE Python (EEG Practical Neuro 2, SS2021) ####
# Aggregate ERPs of several participants

import mne

# Paths and names
preproc_path = '' # enter your data path here
subj_names = ['sub-001', 'sub-002', 'sub-003']


# ... to do before for every subject: save Evoked objects of several conditions within one list
#evoked1 = epochs['cond1'].average()
#evoked2 = epochs['cond2'].average()
#mne.evoked.write_evokeds(preproc_path + '/' '%s_ave.fif' % subj_id, [evoked1, evoked2]) # save as list


# Collect ERPs of all participants
evokeds_group = dict(cond1=[], cond2=[]) # initialize group variable
#evokeds_group = dict(standard=[], deviant=[]) # initialize group variable
for s, subj_id in enumerate(subj_names):
    print('\n\n\nProcessing %s: ... \n' %subj_id)

    evokeds = mne.evoked.read_evokeds(preproc_path + '/' '%s_ave.fif' % subj_id)

    for evoked in evokeds:
        event_type = evoked.comment
        evokeds_group[event_type].append(evoked)


# Grand average (average across all participants)
grand_ave_cond1 = mne.grand_average(evokeds_group['cond1'])
grand_ave_cond2 = mne.grand_average(evokeds_group['cond2'])

# save grand averages
mne.evoked.write_evokeds(preproc_path + '/Grand_ave.fif', [grand_average_cond1, grand_average_cond1])

# load again
grand_ave_load = mne.evoked.read_evokeds(preproc_path + '/Grand_ave.fif')


# a way to obtain a quick visualization with confidence interval:
mne.viz.plot_compare_evokeds(evokeds_group, picks='Fp1', ci=True) # specify the channel of interest with the picks='' argument






