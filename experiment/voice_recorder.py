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
DIR = Path.cwd() / 'experiment' / 'samples' / "laughter_" + vocal_actor

file_name_core = "laughter-" + vocal_actor + "_" + "dist-" + effort_level

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
slab.set_default_samplerate(48828)
config = get_config()
proc_list = config['proc_list']
freefield.initialize('dome', zbus=True, default='play_birec')
freefield.set_logger('WARNING')

playbuflen = 50000
silence = slab.Sound.silence(duration=playbuflen)

freefield.write(tag="playbuflen", value=playbuflen, processors=["RX81", "RX82"])
n_reps = 20

for i in range(n_reps):
    while not freefield.read(tag="response", processor="RP2"):
        time.sleep(0.01)
    print("button was pressed")
    rec_data = freefield.play_and_record(0, silence)
    rec = slab.Binaural(rec_data)

    if numpy.amax(rec.data) > 1:
        print("Audio is clipping")
    else:
        file_name = file_name_core + "_trial-" + str(i + 1) + ".wav"
        rec.write(DIR / file_name)
        print("Done writing:", file_name)
