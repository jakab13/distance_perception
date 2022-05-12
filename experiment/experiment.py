import freefield
from experiment.config import get_config
from experiment.trials import Trials

config = get_config()
proc_list = config['proc_list']
freefield.initialize('dome', zbus=True, device=proc_list)
freefield.set_logger('WARNING')

participant_id = 'blanca'

training = {
    'USOs': Trials(sound_type="USOs", participant_id=participant_id),
    'bark': Trials(sound_type="bark_room-10-30-3", participant_id=participant_id),
    'dunk': Trials(sound_type="dunk_room-10-30-3", participant_id=participant_id),
    'pinknoise': Trials(sound_type="pinknoise_room-10-30-3", participant_id=participant_id),
    'pinknoise_ramped': Trials(sound_type="pinknoise_ramped_room-10-30-3", participant_id=participant_id)
}

training['USOs'].play_control(sound_id=5)
training['USOs'].run(playback_direction='away', scale_type='linear_5', sound_id=5, level=65, n_reps=5)
training['USOs'].run(playback_direction='away', scale_type='linear_5', isi=0.7, level=65)
training['USOs'].run(playback_direction='toward', scale_type='linear_10', sound_id=5, level=65)
training['USOs'].run(playback_direction='toward', scale_type='linear_10', sound_id=5, isi=0.7, level=65)
training['USOs'].run(n_reps=5, record_response=True, level=65)

training['USOs'].run(n_reps=5, level=65, sound_id=5, run_type='experiment')

training['USOs'].play_control(sound_id=5)
training['USOs'].run(playback_direction='away', scale_type='log_5_full', sound_id=5, level=65)
training['USOs'].run(playback_direction='away', scale_type='log_5_full', sound_id=5, isi=0.7, level=65)
training['USOs'].run(playback_direction='toward', scale_type='log_5_full', level=65)
training['USOs'].run(playback_direction='toward', scale_type='log_5_full', sound_id=5, isi=0.7, level=65)
training['USOs'].run(n_reps=5, record_response=True, level=65)

training['bark'].play_control()
training['bark'].run(playback_direction='away', level=65)
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
training['pinknoise'].run(playback_direction='away', scale_type='linear_5', level=70)
training['pinknoise'].run(playback_direction='away', scale_type='linear_5', isi=0.7, level=70)
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
training['pinknoise_ramped'].run(playback_direction='away', scale_type='log_10', isi=0.7, level=65)
training['pinknoise_ramped'].run(playback_direction='toward', level=65)
training['pinknoise_ramped'].run(playback_direction='toward', scale_type='log_10', isi=0.7, level=65)
training['pinknoise_ramped'].run(n_reps=5, record_response=True, level=65)

training['pinknoise_ramped'].play_deviant()
training['pinknoise_ramped'].play_control()

