import freefield
import slab
import pathlib
import os
import time

# ===========================================================

n_reps = 40
isi = 1
filename = 'pinknoise_0'
# filename = 'whisper'

# ============================================================

DIR = pathlib.Path(os.getcwd())  # path for sound and rcx files
default_samplerate = 48828
slab.Signal.set_default_samplerate(default_samplerate)
playbuflen = int(min(isi, 4) * 44100)

proc_list = [['RP2', 'RP2',  DIR / 'data' / 'bi_play_buf.rcx'],
             ['RX81', 'RX8',  DIR / 'data' / 'play_buf.rcx'],
             ['RX82', 'RX8', DIR / 'data' / 'play_buf.rcx']]


file_path = DIR / 'samples'

control_filename = 'AW_A_' + filename + '_long.wav'
dist_1_filename = 'AW_A_' + filename + '_1m.wav'
dist_2_filename = 'AW_A_' + filename + '_2m.wav'
dist_4_filename = 'AW_A_' + filename + '_4m.wav'
dist_8_filename = 'AW_A_' + filename + '_8m.wav'
dist_16_filename = 'AW_A_' + filename + '_16m.wav'

deviant_filename = 'manual_adj_chirp_16.wav'

sound_filenames = [control_filename,
                   # dist_1_filename,
                   dist_2_filename,
                   dist_4_filename,
                   dist_8_filename,
                   dist_16_filename]

def initialise():
    freefield.initialize('dome', zbus=True, device=proc_list)
    freefield.set_logger('WARNING')

def run(filenames):
    deviant_sound = slab.Binaural(file_path / deviant_filename)
    seq = slab.Trialsequence(conditions=filenames, n_reps=n_reps, deviant_freq=0.1)

    for sound_filename in seq:
        # set trigger codes for EEG
        if seq.this_trial == 0:
            # deviant trigger value
            trig_value = 1
        elif seq.this_trial == filenames[0]:
            trig_value = 2
        elif seq.this_trial == filenames[1]:
            trig_value = 3
        elif seq.this_trial == filenames[2]:
            trig_value = 4
        elif seq.this_trial == filenames[3]:
            trig_value = 5
        elif seq.this_trial == filenames[4]:
            trig_value = 6
        freefield.write(tag='trigcode', value=trig_value, processors='RX82')

        if seq.this_trial != 0:
            stimulus = slab.Binaural(file_path / sound_filename)
        else:
            stimulus = deviant_sound
        stimulus.data = stimulus.data[:playbuflen]
        stimulus = stimulus.ramp(duration=0.02)

        # Initialise attributes and data in the .rcx files
        freefield.write(tag="playbuflen", value=playbuflen, processors="RP2")
        freefield.write(tag="data_l", value=stimulus.left.data.flatten(), processors="RP2")
        freefield.write(tag="data_r", value=stimulus.right.data.flatten(), processors="RP2")
        # Playback sound and record participant interaction
        freefield.play()
        print("trial: ", seq.this_n, "soundfile: ", sound_filename)
        prev_response = 0
        while freefield.read(tag="playback", n_samples=1, processor="RP2"):
            curr_response = freefield.read(tag="response", processor="RP2")
            if curr_response > prev_response:
                # attention check trigger value
                trig_value = 7
                freefield.write(tag='trigcode', value=trig_value, processors='RX82')
                freefield.play(proc='RX82')
                print("button was pressed")
            time.sleep(0.01)
            prev_response = curr_response


initialise()
run(sound_filenames)

