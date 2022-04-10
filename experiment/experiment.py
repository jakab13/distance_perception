import os
from experiment.trials import Trials
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

participant_id = 'hannah'

training = {
    'bark': Trials(sound_type="bark", participant_id=participant_id),
    'bum': Trials(sound_type="bum", participant_id=participant_id),
    'chirp': Trials(sound_type="chirp", participant_id=participant_id),
    'dunk': Trials(sound_type="dunk", participant_id=participant_id),
    'pinknoise': Trials(sound_type="pinknoise", participant_id=participant_id),
    'pinknoise_ramped': Trials(sound_type="pinknoise_ramped", participant_id=participant_id),
    'plug': Trials(sound_type="plug", participant_id=participant_id),
    'waterdrop': Trials(sound_type="waterdrop", participant_id=participant_id),
}

training['bark'].play_control()
training['bark'].run(playback_direction='away',  level=65)
training['bark'].run(playback_direction='away', isi=0.7, level=65)
training['bark'].run(playback_direction='toward', level=65)
training['bark'].run(playback_direction='toward', isi=0.7, level=65)
training['bark'].run(n_reps=5, record_response=True, level=65)

training['bum'].play_control()
training['bum'].run(playback_direction='away', isi=2.0, level=65)
training['bum'].run(playback_direction='away', isi=0.7, level=65)
training['bum'].run(playback_direction='toward', level=65)
training['bum'].run(playback_direction='toward', isi=0.7, level=65)
training['bum'].run(n_reps=5, record_response=True, level=65)

training['plug'].play_control()
training['plug'].run(playback_direction='away')
training['plug'].run(playback_direction='away', isi=0.7)
training['plug'].run(playback_direction='toward')
training['plug'].run(playback_direction='toward', isi=0.7)
training['plug'].run(n_reps=5, record_response=True)

training['pinknoise'].play_control()
training['pinknoise'].run(playback_direction='away', level=70)
training['pinknoise'].run(playback_direction='away', isi=0.7, level=70)
training['pinknoise'].run(playback_direction='toward', level=70)
training['pinknoise'].run(playback_direction='toward', isi=0.7, level=70)
training['pinknoise'].run(n_reps=5, record_response=True, level=70)

training['dunk'].play_control()
training['dunk'].run(playback_direction='away', level=70)
training['dunk'].run(playback_direction='away', isi=0.7, level=70)
training['dunk'].run(playback_direction='toward', level=70)
training['dunk'].run(playback_direction='toward', isi=0.7, level=70)
training['dunk'].run(n_reps=5, record_response=True, level=70)

training['pinknoise_ramped'].play_control()
training['pinknoise_ramped'].run(playback_direction='away', level=65)
training['pinknoise_ramped'].run(playback_direction='away', isi=0.7, level=65)
training['pinknoise_ramped'].run(playback_direction='toward', level=65)
training['pinknoise_ramped'].run(playback_direction='toward', isi=0.7, level=65)
training['pinknoise_ramped'].run(n_reps=5, record_response=True, level=65)

training['pinknoise_ramped'].play_deviant()
training['pinknoise_ramped'].play_control()

experiment = Trials(sound_type="pinknoise_ramped")