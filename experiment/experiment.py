import freefield
from experiment.config import get_config
from experiment.trials import Trials

config = get_config()
proc_list = config['proc_list']
freefield.initialize('dome', zbus=True, device=proc_list, )
freefield.set_logger('WARNING')

participant_id = 'paul'

vocalist = Trials(sound_type="vocalist-2", participant_id=participant_id)

vocalist.run(run_type='training', playback_direction='away')
vocalist.run(run_type='training', playback_direction='toward')

vocalist.run(run_type='test', n_reps=2, record_response=True)

vocalist.play_deviant()

vocalist.run(run_type='experiment', n_reps=75)
