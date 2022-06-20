import slab
import freefield
import pathlib
import os
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

DIR = pathlib.Path(os.getcwd()) / 'experiment'


proc_list = [['RP2', 'RP2', DIR / 'data' / 'bi_rec_buf.rcx'],
             ['RX81', 'RX8', DIR / 'data' / 'play_buf.rcx'],
             ['RX82', 'RX8', DIR / 'data' / 'play_buf.rcx']]
freefield.initialize('dome', zbus=True, device=proc_list)

slab.set_default_samplerate(48828)

chirp = slab.Binaural.chirp(from_frequency=200, to_frequency=20000, duration=0.5)
chirp = chirp.ramp(duration=0.01, when='both')

freefield.write(tag="playbuflen", value=chirp.n_samples, processors="RP2")
freefield.write(tag="data_l", value=chirp.left.data.flatten(), processors="RP2")
freefield.write(tag="data_r", value=chirp.right.data.flatten(), processors="RP2")

datal = []
datar = []
for _ in range(20):
    freefield.play()
    datal.append(freefield.read(tag="datal", processor="RP2", n_samples=chirp.n_samples))
    datar.append(freefield.read(tag="datar", processor="RP2", n_samples=chirp.n_samples))
sound_l = slab.Sound(numpy.mean(datal, axis=0))
sound_r = slab.Sound(numpy.mean(datar, axis=0))
sound = slab.Sound(data=numpy.mean((sound_l.data, sound_r.data), axis=0))

# save reference recording
sound.write(DIR / 'headphone_rec.wav')
reference = slab.Sound.read(DIR / 'headphone_rec.wav')

# save target recording
sound.write(DIR / 'in_ear_rec.wav')
target = slab.Sound.read(DIR / 'in_ear_rec.wav')

# create equalizing filterbank
filterbank = slab.Filter.equalizing_filterbank(reference, target, length=1000, bandwidth=1/8, alpha=1.0)
filterbank.tf()
diff = freefield.spectral_range(slab.Sound(data=[target, reference]))

# apply filterbank
equalized_chirp = filterbank.apply(chirp)
freefield.write(tag="data_l", value=equalized_chirp.data, processors="RP2")
freefield.write(tag="data_r", value=equalized_chirp.data, processors="RP2")
datal = []
datar = []
for _ in range(20):
    freefield.play()
    datal.append(freefield.read(tag="datal", processor="RP2", n_samples=chirp.n_samples))
    datar.append(freefield.read(tag="datar", processor="RP2", n_samples=chirp.n_samples))
sound_l = slab.Sound(numpy.mean(datal, axis=0))
sound_r = slab.Sound(numpy.mean(datar, axis=0))
sound = slab.Sound(data=numpy.mean((sound_l.data, sound_r.data), axis=0))

# save filterbank
filterbank.save(DIR / 'in_ear_equalization')
filterbank = slab.Filter.load(DIR / 'in_ear_equalization.npy')

