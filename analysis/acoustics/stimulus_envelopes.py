import slab
import pathlib
import os
from os.path import join
import pyloudnorm as pyln
from pyloudnorm import meter, util, IIRfilter, normalize
import numpy as np
from pydub import AudioSegment

SAMPLERATE = 44100
slab.Signal.set_default_samplerate(SAMPLERATE)
DIR = pathlib.Path(os.getcwd())

VE_directory = DIR / 'experiment' / 'samples' / 'VEs' / 'vocoded-all-original'

def get_file_paths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if not f.startswith('.'):
                yield pathlib.Path(join(dirpath, f))


VE_file_paths = [f for f in get_file_paths(VE_directory)]
VE_sounds = [slab.Binaural(file_path) for file_path in VE_file_paths]
VE_filtered = VE_sounds.copy()
filters = dict()
filters['high_shelf'] = IIRfilter(4.0, 1/np.sqrt(2), 1500.0, SAMPLERATE, 'low_shelf')
filters['high_pass'] = IIRfilter(0.0, 0.5, 38.0, SAMPLERATE, 'high_pass')

meter = pyln.Meter(SAMPLERATE, block_size=0.200)

dist_array = list()
LUFS_diff_array = list()

for idx, sound in enumerate(VE_filtered):
    sound.data = sound.data[:int(0.3*SAMPLERATE)]
    sound = sound.ramp(duration=0.05)
    level_before = sound.level
    numChannels = sound.data.shape[1]
    numSamples = sound.n_samples
    G = [1.0, 1.0, 1.0, 1.41, 1.41] # channel gains
    T_g = 0.2 # 400 ms gating block standard
    Gamma_a = -70.0 # -70 LKFS = absolute loudness threshold
    overlap = 0.75 # overlap of 75% of the block duration
    step = 1.0 - overlap # step size by percentage

    T = numSamples / SAMPLERATE # length of the input in seconds
    numBlocks = int(np.round(((T - T_g) / (T_g * step)))+1) # total number of gated blocks (see end of eq. 3)
    j_range = np.arange(0, numBlocks) # indexed list of total blocks
    z = np.zeros(shape=(numChannels, numBlocks)) # instantiate array - trasponse of input

    for i in range(numChannels):  # iterate over input channels
        for j in j_range:  # iterate over total frames
            l = int(T_g * (j * step    ) * SAMPLERATE)  # lower bound of integration (in samples)
            u = int(T_g * (j * step + 1) * SAMPLERATE)  # upper bound of integration (in samples)
            # caluate mean square of the filtered for each block (see eq. 1)
            z[i,j] = (1.0 / (T_g * SAMPLERATE)) * np.sum(np.square(sound.data[l:u, i]))

    l = [-0.691 + 10.0 * np.log10(np.sum([G[i] * z[i,j] for i in range(numChannels)])) for j in j_range]
    # J_g = [j for j, l_j in enumerate(l) if l_j >= Gamma_a]
    J_g = j_range
    # calculate the average of z[i,j] as show in eq. 7 with blocks above both thresholds
    z_avg_gated = np.nan_to_num(np.array([np.mean([z[i,j] for j in J_g]) for i in range(numChannels)]))
    LUFS = meter.integrated_loudness(sound.data)
    my_LUFS = -0.691 + 10.0 * np.log10(np.sum([G[i] * z_avg_gated[i] for i in range(numChannels)]))
    LUFS_diff = LUFS - my_LUFS
    file_path = VE_file_paths[idx]
    distance = file_path.name[file_path.name.find('dist-') + len('dist-'):file_path.name.rfind('_try')]
    distance = int(distance)
    dist_array.append(distance)
    LUFS_diff_array.append(LUFS_diff)
    print(LUFS_diff)
    # print("LUFS:", LUFS, "my_LUFS:", my_LUFS)
    sound.data = pyln.normalize.loudness(sound.data, my_LUFS, -30+LUFS_diff)
    output_filename = VE_directory.parent / "vocoded-all-reconstructed-test" / str(VE_file_paths[idx].stem + "_reconstructed.wav")
    sound.write(output_filename, normalise=False)
