import slab
import freefield
import pathlib
import os
import time

DIR = pathlib.Path(os.getcwd())  # path for sound and rcx files
default_samplerate = 48828
slab.Signal.set_default_samplerate(default_samplerate)

file_path = DIR / 'samples'
filename = 'pinknoise_0'
dist_2_filename = 'AW_A_' + filename + '_2m.wav'
dist_4_filename = 'AW_A_' + filename + '_4m.wav'
dist_8_filename = 'AW_A_' + filename + '_8m.wav'
dist_16_filename = 'AW_A_' + filename + '_16m.wav'
sound_filenames = [dist_2_filename,
                   dist_4_filename,
                   dist_8_filename,
                   dist_16_filename]

file_path = DIR / 'samples'
filename = 'pinknoise'
dist_2_filename = filename + '_2.wav'
dist_4_filename =  filename + '_4.wav'
dist_8_filename =  filename + '_8.wav'
dist_16_filename =  filename + '_16.wav'
sound_filenames = [dist_2_filename,
                   dist_4_filename,
                   dist_8_filename,
                   dist_16_filename]


proc_list = [['RP2', 'RP2',  DIR / 'data' / 'bi_play_buf.rcx'],
             ['RX81', 'RX8',  DIR / 'data' / 'play_buf.rcx'],
             ['RX82', 'RX8', DIR / 'data' / 'play_buf.rcx']]

def initialise():
    freefield.initialize('dome', zbus=True, device=proc_list)
    freefield.set_logger('WARNING')

def run(filenames):
    seq = slab.Trialsequence(conditions=filenames, n_reps=25)
    #data frame to hold response
    resp = pd.DataFrame(columns=['stimulus', 'response', 'match'], index=range(seq.n_trials))

    for sound_filename in seq:

        # get stimulus
        curr_dist = seq.trials[seq.this_n]
        resp.at[seq.this_n, 'stimulus'] = curr_dist
        stimulus = slab.Binaural(file_path / sound_filename)
        print("trial: ", seq.this_n)

        # load data to buffer
        stimulus.data = stimulus.data[:playbuflen]
        stimulus = stimulus.ramp(duration=0.02)
        freefield.write(tag="playbuflen", value=playbuflen, processors="RP2")
        freefield.write(tag="data_l", value=stimulus.left.data.flatten(), processors="RP2")
        freefield.write(tag="data_r", value=stimulus.right.data.flatten(), processors="RP2")

        # Playback sound and record participant interaction
        freefield.play()
        while response == 0:
            response = freefield.read(tag="response", proc="RP2")
            time.sleep(.1)
        response = freefield.read(tag="response", proc="RP2")
        resp.at[seq.this_n, 'response'] = response # write response to data frame

        if response == curr_dist: # check whether response matches target
            seq.add_response(1)
            resp.at[seq.this_n, 'match'] = 1
            else:
            seq.add_response(0)
            resp.at[seq.this_n, 'match'] = 0
        time.sleep(2)

if __name__ == "__main__":
    initialise()
    run(sound_filenames)


""" 
seq = slab.Trialsequence(conditions=sound_filenames, n_reps=1)
resp = []
for filename in seq:
    dist = seq.trials[seq.this_n]
    stimulus = slab.Binaural(file_path / sound_filename)
    print("trial: ", seq.this_trial)
    stimulus.play()
    response = input("Please chose from 1, 2, 3, 4:\n")
    if int(response) == dist:
        seq.add_response(1)
        resp.append[1]
    else:
        seq.add_response(0)
        resp.append[0]
seq.save_pickle('test.pkl', clobber=True)
"""