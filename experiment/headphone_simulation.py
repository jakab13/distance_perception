import copy
import freefield
import numpy as np
import slab
import pathlib
import os
import time
from os.path import join
import random
from scipy import signal

# ===========================================================

n_reps = 60
isi = 1.5
filename = 'laughter_pilot'
room = '10-30-3'
level = 62

# ============================================================

# DIR = pathlib.Path(__file__).parent.absolute()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DIR = pathlib.Path(os.getcwd())
default_samplerate = 48828
slab.set_default_samplerate(44100)
playbuflen = int(min(isi, 4) * default_samplerate)

proc_list = [['RP2', 'RP2',  DIR / 'experiment' / 'data' / 'bi_play_buf.rcx'],
             ['RX81', 'RX8',  DIR / 'experiment' / 'data' / 'play_buf.rcx'],
             ['RX82', 'RX8', DIR / 'experiment' / 'data' / 'play_buf.rcx']]

folder_path = DIR / 'experiment' / 'samples' / filename / 'a_weighted'

def abs_file_paths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            if not f.startswith('.'):
                yield pathlib.Path(join(dirpath, f))


all_file_paths = [f for f in abs_file_paths(folder_path)]

dist_1_filenames = [f for f in all_file_paths if "dis1" in f.name]
dist_2_filenames = [f for f in all_file_paths if "dis2" in f.name]
dist_3_filenames = [f for f in all_file_paths if "dis3" in f.name]
dist_4_filenames = [f for f in all_file_paths if "dis4" in f.name]
dist_5_filenames = [f for f in all_file_paths if "dis5" in f.name]

# control_filename = 'AW_A_' + filename + '_control.wav'
# dist_1_filename = 'AW_A_' + filename + '_dist-20.wav'
# dist_2_filename = 'AW_A_' + filename + '_dist-200.wav'
# dist_4_filename = 'AW_A_' + filename + '_dist-1000.wav'
# dist_8_filename = 'AW_A_' + filename + '_dist-2000.wav'
# dist_16_filename = 'AW_A_' + filename + '_dist-1600.wav'

deviant_filepath = DIR / 'experiment' / 'samples' / 'chirp_room-10-30-3' / 'a_weighted' \
                   / 'AW_A_chirp_room-10-30-3_control.wav'
# control_filepath = file_path / control_filename

sound_file_categories = [
    # control_filename,
    dist_1_filenames,
    dist_2_filenames,
    dist_3_filenames,
    dist_4_filenames,
    dist_5_filenames
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

stimulus_categories = [[] for i in range(len(sound_file_categories))]
for i in range(len(sound_file_categories)):
    stimulus_categories[i] = []
    for sound_file_name in sound_file_categories[i]:
        sound = slab.Binaural(sound_file_name)
        stimulus_categories[i].append(sound)
deviant_sound = slab.Binaural(deviant_filepath)
stimulus_categories.insert(0, [deviant_sound])

for i in range(len(stimulus_categories)):
    stimulus_category = stimulus_categories[i]
    for j, binaural in enumerate(stimulus_category):
        stimulus = stimulus_category[j]
        out = copy.deepcopy(stimulus)
        stimulus_length = len(out.data)
        if stimulus_length >= playbuflen:
            out.data = out.data[:playbuflen]
            out = out.ramp(duration=0.05)
            out.data = signal.resample(out.data, int(out.duration * default_samplerate))
            out.samplerate = default_samplerate
        else:
            silence_length = playbuflen - stimulus_length
            silence = slab.Sound.silence(duration=silence_length, samplerate=default_samplerate)
            left = slab.Sound(out.data[:, 0], samplerate=default_samplerate)
            right = slab.Sound(out.data[:, 1], samplerate=default_samplerate)
            left = left.ramp(duration=0.05)
            right = right.ramp(duration=0.05)
            left = slab.Sound.sequence(left, silence)
            right = slab.Sound.sequence(right, silence)
            out = slab.Binaural([left, right])
            out.data = out.data
            out.data = signal.resample(out.data, int(out.duration * default_samplerate))
            out.samplerate = default_samplerate
        stimulus_categories[i][j] = out
        print("done sampling")


seq = slab.Trialsequence(conditions=[1, 2, 3, 4, 5], n_reps=n_reps, deviant_freq=0.1)
for distance_group in seq:
    sound = random.choice(stimulus_categories[distance_group])
    if distance_group == 0:
        sound.level = level - 5
    else:
        sound.level = level
    # sound_filename = sound_filenames[distance_group - 1]
    # set trigger codes for EEG
    if distance_group == 0:
        # deviant
        trig_value = 1
        sound_filename = 'deviant'
    elif distance_group == 1:
        # control
        trig_value = 2
    elif distance_group == 2:
        trig_value = 3
    elif distance_group == 3:
        trig_value = 4
    elif distance_group == 4:
        trig_value = 5
    elif distance_group == 5:
        trig_value = 6
    freefield.write(tag='trigcode', value=trig_value, processors='RX82')
    # Initialise attributes and data in the .rcx files
    freefield.write(tag="playbuflen", value=playbuflen, processors="RP2")
    freefield.write(tag="data_l", value=sound.left.data.flatten(), processors="RP2")
    freefield.write(tag="data_r", value=sound.right.data.flatten(), processors="RP2")
    # Playback sound and record participant interaction
    print("playing trial:", seq.this_n, "distance group:", distance_group)
    freefield.play()
    button_trig(7)

