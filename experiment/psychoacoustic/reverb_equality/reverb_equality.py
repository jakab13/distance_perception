import os
import random
import slab
import matplotlib.pyplot as plt
import pathlib
import numpy

slab.set_default_samplerate(44100)
results_folder = "/Users/jakabpilaszanovich/Documents/GitHub/distance_perception/experiment/psychoacoustic/reverb_equality/results"
results_file = slab.ResultsFile(subject='marc', folder=results_folder)

folder_path = pathlib.Path("/Users/jakabpilaszanovich/Documents/GitHub/distance_perception") / "experiment" / 'samples' / "clap2" / "simulated"

file_names = [f for f in os.listdir(folder_path) if not f.startswith('.')]
sounds_by_t60 = {}
for file_name in file_names:
    t60 = file_name[file_name.find('_t60-') + len('_t60-'):file_name.rfind('_dist')]
    sounds_by_t60[float(t60)] = slab.Sound(folder_path / file_name)
t60s = sorted(list(sounds_by_t60.keys()))
max_length = max([sound.n_samples for t60, sound in sounds_by_t60.items()])

global fig_key
global t60_index
def _on_key(event):
    global fig_key
    global t60_index
    fig_key = event.key
    if fig_key == "1" and t60_index < len(t60s) - 1:
        t60_index += 1
    elif fig_key == "9" and t60_index > 1:
        t60_index -= 1

ref_t60 = 1.5
ref_sound = sounds_by_t60[ref_t60]
tone = slab.Sound.tone(frequency=500, duration=0.05)

seq = slab.Trialsequence(conditions=5)
for trial in seq:
    fig_key = None
    seed_t60 = random.choice([i for i in t60s if i != ref_t60])
    t60_index = t60s.index(seed_t60)
    fig = plt.figure()
    _ = fig.canvas.mpl_connect('key_press_event', _on_key)
    ref_sound.play()
    while fig_key != "c":
        fig_key = None
        plt.pause(0.01)
        print("Updated t60:", str(t60s[t60_index]) + "s", "(" + str(fig_key) + ")")
        curr_sound = sounds_by_t60[t60s[t60_index]]
        reversed_sound = slab.Sound(numpy.flipud(curr_sound.data), samplerate=curr_sound.samplerate)
        reversed_sound.play()
    results_file.write(t60s[t60_index], tag="reverse_t60")
    plt.close()
