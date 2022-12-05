import slab
import freefield
import numpy
import time
import os
from pathlib import Path
from experiment.config import get_config

#######################################################################

vocal_actor = "jakab"
effort_level = "1"

#######################################################################

# DIR = pathlib.Path(__file__).parent.absolute()
DIR = Path.cwd() / 'experiment' / 'samples' / str("laughter_" + vocal_actor)

file_name_core = "laughter-" + vocal_actor + "_" + "dist-" + effort_level

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
samplerate = 48828
slab.set_default_samplerate(samplerate)
config = get_config()
freefield.initialize('dome', zbus=True, device=['RP2', 'RP2', 'D:/Projects/distance_perception/experiment/data/button_rec.rcx'])
freefield.set_logger('WARNING')


ready_to_record = True
counter = 0
while ready_to_record:
    print('hold button to start recording')
    while not freefield.read(tag="response", processor="RP2"):
        time.sleep(0.01)
    print("button was pressed")
    start_time = time.time()
    while freefield.read(tag="response", processor="RP2"):
        time.sleep(0.1)
    end_time = time.time()
    rec_length = round((end_time - start_time) * samplerate)
    rec_l = freefield.read(tag="data_l", processor="RP2", n_samples=rec_length)
    rec_r = freefield.read(tag="data_r", processor="RP2", n_samples=rec_length)
    rec = slab.Binaural([rec_l, rec_r])
    # file_name = file_name_core + "_trial-" + str(counter + 1) + ".wav"
    # rec.write(DIR / file_name)
    # print("Done writing:", file_name)
    ready_to_record = 'y' == input("Are you ready to record again? (y/n)")
    counter += 1