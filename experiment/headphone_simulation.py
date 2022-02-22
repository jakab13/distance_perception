import freefield
import numpy as np
import slab
import pathlib
import os
import time
from experiment.config import get_config
from experiment.load import load_sounds

# ===========================================================

n_reps = 50
isi = 1.5
sound_type = 'pinknoise'
# sound_type = 'whisper'
# sound_type = 'bark'

# ============================================================

config = get_config()
DIR = pathlib.Path(os.getcwd())  # path for sound and rcx files
default_samplerate = 48828
slab.set_default_samplerate(default_samplerate)
playbuflen = int(min(isi, 4) * 44100)

def button_trig(trig_value):
    prev_response = 0
    while freefield.read(tag="playback", n_samples=1, processor="RP2"):
        curr_response = freefield.read(tag="response", processor="RP2")
        if curr_response > prev_response:
            print("button was pressed")
            freefield.write(tag='trigcode', value=trig_value, processors='RX82')
            print("trigcode was set to:", trig_value)
            freefield.play(proc='RX82')
        time.sleep(0.01)
        prev_response = curr_response

freefield.initialize('dome', zbus=True, device=config['proc_list'])
freefield.set_logger('WARNING')

loaded_sound = load_sounds()
distances = [0.2, 2, 8, 18]

stimuli = [slab.Binaural(file_path / sound_filename) for sound_filename in sound_filenames]
deviant_sound = slab.Binaural(deviant_filepath)
stimuli.insert(0, deviant_sound)
for idx, stimulus in enumerate(stimuli):
    stimulus_length = len(stimuli[idx].data)
    if stimulus_length >= playbuflen:
        stimuli[idx].data = stimuli[idx].data[:playbuflen]
    else:
        silence_length = playbuflen - stimulus_length
        silence = slab.Sound.silence(duration=silence_length, samplerate=stimuli[idx].samplerate)
        left = slab.Sound(stimuli[idx].data[:, 0], samplerate=stimuli[idx].samplerate)
        right = slab.Sound(stimuli[idx].data[:, 1], samplerate=stimuli[idx].samplerate)
        left = slab.Sound.sequence(left, silence)
        right = slab.Sound.sequence(right, silence)
        stimuli[idx] = slab.Binaural([left, right])
    stimuli[idx] = stimuli[idx].ramp(duration=0.02)

seq = slab.Trialsequence(conditions=len(stimuli), n_reps=n_reps, deviant_freq=0.05)
for stimulus_id in seq:
    sound = stimuli[stimulus_id - 1]
    current_stimulus_id = seq.this_trial
    sound_filename = sound_filenames[current_stimulus_id - 2]
    # set trigger codes for EEG
    if current_stimulus_id == 1:
        # deviant
        trig_value = 1
        sound_filename = deviant_filename
    elif current_stimulus_id == 2:
        # control
        trig_value = 2
    elif current_stimulus_id == 3:
        trig_value = 3
    elif current_stimulus_id == 4:
        trig_value = 4
    elif current_stimulus_id == 5:
        trig_value = 5
    elif current_stimulus_id == 6:
        trig_value = 6
    freefield.write(tag='trigcode', value=trig_value, processors='RX82')
    # Initialise attributes and data in the .rcx files
    freefield.write(tag="playbuflen", value=playbuflen, processors="RP2")
    freefield.write(tag="data_l", value=sound.left.data.flatten(), processors="RP2")
    freefield.write(tag="data_r", value=sound.right.data.flatten(), processors="RP2")
    # Playback sound and record participant interaction
    print("playing trial:", seq.this_n, "file:", sound_filename)
    freefield.play()
    button_trig(7)


