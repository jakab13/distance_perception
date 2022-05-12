import freefield
import numpy as np
import slab
import pathlib
import os
import time
from pprint import pprint

# ===========================================================

n_reps = 60
isi = 1.5
filename = 'pinknoise_ramped_room-10-30-3'
room = '10-30-3'
level = 65
# filename = 'clicktrain'
# filename = 'whisper'
# filename = 'bark'

# ============================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DIR = pathlib.Path(os.getcwd())
# DIR = pathlib.Path(__file__).parent.absolute()
default_samplerate = 48828
slab.set_default_samplerate(default_samplerate)
playbuflen = int(min(isi, 4) * 44100)

proc_list = [['RP2', 'RP2',  DIR / 'experiment' / 'data' / 'bi_play_buf.rcx'],
             ['RX81', 'RX8',  DIR / 'experiment' / 'data' / 'play_buf.rcx'],
             ['RX82', 'RX8', DIR / 'experiment' / 'data' / 'play_buf.rcx']]

file_path = DIR / 'experiment' / 'samples' / filename / 'a_weighted'

control_filename = 'AW_A_' + filename + '_control.wav'
dist_1_filename = 'AW_A_' + filename + '_dist-20.wav'
dist_2_filename = 'AW_A_' + filename + '_dist-200.wav'
dist_4_filename = 'AW_A_' + filename + '_dist-1000.wav'
dist_8_filename = 'AW_A_' + filename + '_dist-2000.wav'
# dist_16_filename = 'AW_A_' + filename + '_dist-1600.wav'

deviant_filepath = DIR / 'experiment' / 'samples' / 'chirp_room-10-30-3' / 'a_weighted' \
                   / 'AW_A_chirp_room-10-30-3_control.wav'
control_filepath = file_path / control_filename

sound_filenames = [
    control_filename,
    dist_1_filename,
    dist_2_filename,
    dist_4_filename,
    dist_8_filename,
    # dist_16_filename
]

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

freefield.initialize('dome', zbus=True, device=proc_list)
freefield.set_logger('WARNING')

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

seq = slab.Trialsequence(conditions=[1, 2, 3, 4, 5], n_reps=n_reps, deviant_freq=0.1)
for stimulus_id in seq:
    print(stimulus_id)
    sound = stimuli[stimulus_id]
    if stimulus_id == 1 or stimulus_id == 0:
        sound.level = level - 5
    else:
        sound.level = level
    sound_filename = sound_filenames[stimulus_id - 1]
    # set trigger codes for EEG
    if stimulus_id == 0:
        # deviant
        trig_value = 1
        sound_filename = 'deviant'
    elif stimulus_id == 1:
        # control
        trig_value = 2
    elif stimulus_id == 2:
        trig_value = 3
    elif stimulus_id == 3:
        trig_value = 4
    elif stimulus_id == 4:
        trig_value = 5
    elif stimulus_id == 5:
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


