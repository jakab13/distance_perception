import os
import slab
import pathlib
import numpy as np
import matplotlib.pyplot as plt

slab.set_default_samplerate(44100)

rec_folder_path = pathlib.Path("/Users/jakabpilaszanovich/Documents/GitHub/distance_perception") / "experiment" / 'samples' / "distance_plasticity" / "cathedral_recordings"
rec_file_names = [f for f in os.listdir(rec_folder_path) if not f.startswith('.')]
sim_folder_path = pathlib.Path("/Users/jakabpilaszanovich/Documents/GitHub/distance_perception") / "experiment" / 'samples' / "distance_plasticity" / "clap2" / "simulated"
sim_file_names = [f for f in os.listdir(sim_folder_path) if not f.startswith('.')]


def average_sounds(sound_list):
    rec_avg = slab.Sound.silence(duration=np.amax([s.n_samples for s in sound_list]))
    for sound in sound_list:
        rec_avg.data[:sound.n_samples] += sound.data
    rec_avg.data = rec_avg.data / len(sound_list)
    rec_avg.data = rec_avg.data - 0.22  # DC offset???
    return rec_avg


def measure_rt60(h, fs=44100, decay_db=60, plot=False, rt60_tgt=None):
    """
    Analyze the RT60 of an impulse response. Optionaly plots some useful information.

    Parameters
    ----------
    `h`: array_like
        The impulse response.
    `fs`: float or int, optional
        The sampling frequency of h (default to 1, i.e., samples).
    `decay_db`: float or int, optional
        The decay in decibels for which we actually estimate the time. Although
        we would like to estimate the RT60, it might not be practical. Instead,
        we measure the RT20 or RT30 and extrapolate to RT60.
    `plot`: bool, optional
        If set to ``True``, the power decay and different estimated values will
        be plotted (default False).
    `rt60_tgt`: float
        This parameter can be used to indicate a target RT60 to which we want
        to compare the estimated value.
    """

    h = np.array(h)
    fs = float(fs)

    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    # -5 dB headroom
    i_5db = np.min(np.where(-5 - energy_db > 0)[0])
    e_5db = energy_db[i_5db]
    t_5db = i_5db / fs

    # after decay
    i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs

    # compute the decay time
    decay_time = t_decay - t_5db
    est_rt60 = (60 / decay_db) * decay_time

    if plot:
        import matplotlib.pyplot as plt

        # Remove clip power below to minimum energy (for plotting purpose mostly)
        energy_min = energy[-1]
        energy_db_min = energy_db[-1]
        power[power < energy[-1]] = energy_min
        power_db = 10 * np.log10(power)
        power_db -= np.max(power_db)

        # time vector
        def get_time(x, fs):
            return np.arange(x.shape[0]) / fs

        T = get_time(power_db, fs)

        # plot power and energy
        plt.plot(get_time(energy_db, fs), energy_db, label="Energy")

        # now the linear fit
        # plt.plot([0, est_rt60], [0, -60], "--", label="Linear Fit (RT20)")
        plt.plot(T, np.ones_like(T) * -decay_db, "--", label=f"-{decay_db} dB")
        # plt.vlines(
        #     est_rt60, energy_db_min, 0, linestyles="dashed", label="Estimated RT60"
        # )
        # if rt60_tgt is not None:
        #     plt.vlines(rt60_tgt, energy_db_min, 0, label="Target RT60")
        plt.xlim(-0.5, max(T) + 0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Sound energy (dB)")
        plt.title("Estimated RT20")
        plt.legend()
        plt.show()
    return est_rt60


recs = [slab.Sound(rec_folder_path / f) for f in rec_file_names]
rec_avg = average_sounds(recs)

rec_fbank = slab.Filter.cos_filterbank(length=rec_avg.n_samples, bandwidth=1) # make the filter bank
rec_subbands = rec_fbank.apply(rec_avg)
rec_envs = rec_subbands.envelope()

for i in range(1, rec_subbands.n_channels):
    rec_subband = rec_subbands[:, i]
    rt60 = measure_rt60(rec_subband, decay_db=15, plot=True)
    print(rt60)

sim_sounds = [slab.Sound(sim_folder_path / f) for f in sim_file_names]
sim_sound = sim_sounds[0]
sim_sound.data = sim_sound.data.mean(axis=1)
sim_sound.data = sim_sound.data[:, np.newaxis]

sim_fbank = slab.Filter.cos_filterbank(length=sim_sound.n_samples, bandwidth=1) # make the filter bank
sim_subbands = sim_fbank.apply(sim_sound)
sim_envs = sim_subbands.envelope()

for i in range(1, sim_subbands.n_channels):
    sim_subband = sim_subbands[:, i]
    rt60 = measure_rt60(sim_subband, decay_db=15, plot=True)
    print(rt60)