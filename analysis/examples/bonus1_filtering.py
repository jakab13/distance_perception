# In this script we will take a closer look at the effect of filtering on the data
# For this we will not use real EEG recordings but rather simulate them.
# The advantage of simulated data is that we know what they are made of so we can tell apart signal and artifacts

import numpy
import scipy.signal
from matplotlib import pyplot as plt

#Lets assume we have an EEG signal of a certain duration:

duration = 20  # in seconds
samplerate = 500  # in Hz
nyq = samplerate/2
time = numpy.linspace(0, duration, duration*samplerate)

# The signal has three different sources of noise (meaning things that we are not interested in):
# the 50 Hz signal from the power line, a slow drift and random noise

power_line_frequency = 50.  # in Hz
drift_frequency = .1

power_line = numpy.sin(2. * numpy.pi * power_line_frequency * time)*15
drift = numpy.sin(2. * numpy.pi * drift_frequency * time)*5
noise = numpy.random.rand(duration*samplerate)*3

# We also have evoked activity. We will model this as a 100 millisecond square pulse whenever a stimulus is played:
# the stimuli will appear every 2 seconds with a jitter of 1 second
evoked = numpy.zeros(duration*samplerate)
events = numpy.linspace(1, 18, 10) + numpy.random.uniform(-.5, .5, 10)
for e in events:
    idx = numpy.argmin(numpy.abs(time-e))
    evoked[idx:idx+50] = 1

# plot the different signals:
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(time, noise)
ax[0, 0].set_title("random noise")
ax[0, 1].plot(time, power_line)
ax[0, 1].set_title("power line")
ax[1, 0].plot(time, drift)
ax[1, 0].set_title("electrode drift")
ax[1, 1].plot(time, evoked)
ax[1, 1].set_title("evoked activity")
plt.show()

# now we combine the signal:
signal = power_line+drift+noise+evoked
plt.plot(time, signal)
plt.show()

# and epoch the data:
tmin = -.1
tmax = 1
epochs = numpy.zeros([int(numpy.abs(tmin-tmax)*samplerate), len(events)])
epoch_times = numpy.linspace(tmin, tmax, int(numpy.abs(tmin-tmax)*samplerate))
for i, e in enumerate(events):
    idx = numpy.argmin(numpy.abs(time-e))
    epochs[:, i] = signal[idx+int(tmin*samplerate):idx+int(tmax*samplerate)]

plt.plot(epoch_times, epochs.mean(axis=1))
# you can get a small hint of evoked activity but it is really weak. Let's try filtering to make it clearer.

# first we have to design the filter, for example a bandpass:
order = 10  # filter order determines how steep the cutoff will be
f1, f2 = 1, 40  # the filter's lower and higher cutoff
numtaps = 600  # length of the filter, also knows as "taps"

band = scipy.signal.firwin(numtaps, [f1, f2], pass_zero=False, fs=samplerate, window="hamming")
freqs, magnitude = scipy.signal.freqz(band, fs=samplerate)  # the filters magnitude response

# it's a good practice to always plot the impulse and frequency response of the filter
fig, ax = plt.subplots(2)
ax[0].plot(band)
ax[0].set_title("Impulse Response")
ax[1].plot(freqs, 20*numpy.log10(magnitude).real)
ax[1].set_title("Magnitude Response")

# Applying a filter means convolving the filters impulse response with the signal
signal_filt = numpy.convolve(band, signal)
# instead you could also use scipy's filtfilt function to apply the filter:
signal_filt = scipy.signal.filtfilt(band, [1], signal)
# compare the signals and see how they differ !

# Now we can epoch the filtered signal and plot the result
epochs = numpy.zeros([int(numpy.abs(tmin-tmax)*samplerate), len(events)])
for i, e in enumerate(events):
    idx = numpy.argmin(numpy.abs(time-e))
    epochs[:, i] = signal_filt[idx+int(tmin*samplerate):idx+int(tmax*samplerate)]

plt.plot(epoch_times, epochs.mean(axis=1))
plt.vlines(0, ymin=epochs.mean(axis=1).min(), ymax=epochs.mean(axis=1).max(), color="black", linestyles="--")

# since we are using a simulation, we can also apply the filter to each component individually to see it's effect:
power_line_filt = scipy.signal.filtfilt(band, [1], power_line)
noise_filt = scipy.signal.filtfilt(band, [1], noise)
drift_filt = scipy.signal.filtfilt(band, [1], drift)
evoked_filt = scipy.signal.filtfilt(band, [1], evoked)

# plot the different signals:
fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(time, noise_filt)
ax[0, 0].set_title("random noise")
ax[0, 1].plot(time, power_line_filt)
ax[0, 1].set_title("power line")
ax[1, 0].plot(time, drift_filt)
ax[1, 0].set_title("electrode drift")
ax[1, 1].plot(time, evoked_filt)
ax[1, 1].set_title("evoked activity")
plt.show()

# Now it's you turn: try out different filters on the data and see how the data is affected by them.
# you can also change the composition of the signal, for example you can use a "brain signal" that has a positive
# and a negative phase or one that is not square shaped. If you are looking for some theoretical background, have
# a look at the paper Filters: When, Why, and How (Not) to Use Them.