import mne
from matplotlib import pyplot as plt

# The data for the experiment comes from a study where they showed different words to the subject.
# You can find the paper here: https://journals.sagepub.com/doi/10.1177/0956797615603934
path = mne.datasets.kiloword.data_path() + '/kword_metadata-epo.fif'
epochs = mne.read_epochs(path)
time_unit = dict(time_unit="s")

# The words and some metrics describing them are stored in the "metadata" attribute of the epochs
print(epochs.metadata)

# We will take a look at the number of letters in a word by splitting them into
# short (below median) and long (above median) words
name = "NumberOfLetters"
median_value = str(epochs.metadata[name].median())
long_words = epochs[name + " > " + median_value]
short_words = epochs[name + " < " + median_value]

# First, we calculate and plot the difference between the two conditions we are testing.
# Discuss the plot: what are your expectations for running statistical tests on this data?
evoked = mne.combine_evoked([long_words.average(), short_words.average()], weights=[1, -1])
evoked.plot_joint(title="Long vs. short words", ts_args=time_unit, topomap_args=time_unit)


# Now we will perform a permutation cluster test on the data. If you want to understand the procedure in detail, you
# can read the paper "Nonparametric statistical testing of EEG- and MEG-data", but in short the approach is:
# 1. compute the difference between the conditions
# 2. assume the null hypothesis that the conditions are the same, meaning that the difference we are observing
# is purely by chance
# 3. randomly assign epochs to one condition (if they really were the same this should not matter)
# 4. compute the difference between the randomly assigned conditions
# --> repeat steps 3 & 4 again and again (like 1000 times or so)
# If the difference in our conditions was not merely a product of chance then the differences we observe in the
# randomly assigned conditions should rarely be as large as the original one.

# we will run the analysis across all channels so we need the adjacency matrix
adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, "eeg")
plt.matshow(adjacency.toarray())  # take a look at the matrix

# the permutation test expects the data to be in the shape: observations × time × space.
# in our case, the observations are single epochs. You could also test across multiple subjects. In this case,
# one observation would be the evoked response of one subject.
X = [long_words.get_data().transpose(0, 2, 1),
     short_words.get_data().transpose(0, 2, 1)]

# Calculate statistical thresholds. For times sake we are only doing 100 permutations but in a "real" analysis
# you would at least do 1000.
t_obs, clusters, cluster_pv, h0 = mne.stats.spatio_temporal_cluster_test(
    X, threshold=dict(start=.2, step=.2), adjacency=adjacency, n_permutations=100)

# We can see the number of significant points in the data by summing all the values in the test statistic which
# have a value smaller .05
significant_points = cluster_pv.reshape(t_obs.shape).T < .05
print(str(significant_points.sum()) + " points selected by TFCE ...")

# We can also visualize them in a plot (only significant points are shown in color):
evoked.plot_image(mask=significant_points, show_names="all", titles=None, **time_unit)

# Now it's your turn. The words used in the experiment have more features than just their length. Discuss how
# these other features might be reflected in the EEG signal and then go ahead and run the test.

