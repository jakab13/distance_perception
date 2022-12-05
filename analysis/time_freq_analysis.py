import numpy as np
import mne
import matplotlib.pyplot as plt

sampling_freq = raw.info['sfreq']


def tf_analysis(data, plot=0):    # TF analysis
    # wavelet parameters:
    num_frex = 101
    min_freq = 1
    max_freq = 100
    frex = np.linspace(min_freq, max_freq, num_frex) # specify frequency range to analyze
    num_cycles = [50] # number of cycles (tradeoff between time and freq res)
    time = np.linspace(-2, 2, int(sampling_freq*4)+1)
    half_wave = int((len(time)-1)/2)
    # FFT parameters
    nKern = len(time)
    nData = len(data.flatten())
    nConv = nKern+nData-1
    baseline_idx = [0, 2000] # range of initial data points to use as baseline (per frequency)
    # initialize output time-frequency data
    tf_data = np.zeros([len(num_cycles), len(frex), data.shape[0]])
    # FFT of data(doesnt change on frequency iteration)
    dataX = np.fft.fft(data.flatten(), n=nConv)
    # loop over cycles and frequencies
    for c, cycles in enumerate(num_cycles):
        for f, freq in enumerate(frex):
            # create complex morlet wavelet and get its FFT
            s = cycles/(2*np.pi*freq)
            cmw = np.exp(2j*np.pi*freq*time) * np.exp(-time ** 2/(2*s ** 2))
            cmwX = np.fft.fft(cmw, n=nConv)
            cmwX = cmwX/max(cmwX)
            signal = np.fft.ifft(cmwX*dataX)  # run convolution
            signal = signal[half_wave:-half_wave]  # trim edges
            tf_data[c, f, :] = np.abs(signal)**2
    # Compute the baseline for each number of cycles and each frequency
    baseline_data = tf_data[:, :, baseline_idx[0]:baseline_idx[1]].mean(axis=2)
    tf_data_norm = 10*np.log10(tf_data / baseline_data[:, :, np.newaxis])
    if plot:
        plot_tf_data(tf_data_norm, frex)
    return tf_data_norm


def plot_tf_data(tf_data, frex):  # input: normalized time frequency data (time_series x frequencies)
    time = np.arange(0, len(tf_data)/sampling_freq, 1/sampling_freq)
    t_crop = 0.1
    n_crop = int(t_crop * sampling_freq)
    x, y = np.meshgrid(time[n_crop:-n_crop], frex)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    for i, cycles in enumerate(num_cycles):
        z = tf_data[i,:, n_crop:-n_crop]
        cax = ax.contour(x, y, z, linewidths=0.3, colors="k", norm=colors.Normalize())
        cax = ax.contourf(x, y, z, norm=colors.Normalize(), cmap=plt.cm.jet)
        ax.set_title("wavelet with %s cycles"%(cycles))
    cbar = fig.colorbar(cax)
    plt.show()


