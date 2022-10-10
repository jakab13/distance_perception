from experiment.trials import Trials, Experiment

experiment = Experiment()
experiment.initialise()

participant_id = 'pilot'

bark = Trials(sound_type="bark", participant_id=participant_id)
bum = Trials(sound_type="bum", participant_id=participant_id)
dunk = Trials(sound_type="dunk", participant_id=participant_id)
USOs = Trials(sound_type="USOs", participant_id=participant_id)

bark.run(stage='training', scale_type='USO_scale', playback_direction='away', level=68)
bark.run(stage='training', scale_type='USO_scale', playback_direction='toward', level=68)
dunk.run(stage='training', scale_type='USO_scale', playback_direction='away', level=68)
dunk.run(stage='training', scale_type='USO_scale', playback_direction='toward', level=68)
bum.run(stage='training', scale_type='USO_scale', playback_direction='away', level=68)
bum.run(stage='training', scale_type='USO_scale', playback_direction='toward', level=68)

USOs.run(stage='training', scale_type='USO_scale', playback_direction='away', sound_id=2)
USOs.run(stage='training', scale_type='USO_scale', playback_direction='toward', sound_id=2)
USOs.run(stage='training', scale_type='USO_scale', playback_direction='away', sound_id=21)
USOs.run(stage='training', scale_type='USO_scale', playback_direction='toward', sound_id=21)
USOs.run(stage='training', scale_type='USO_scale', playback_direction='away', sound_id='random')
USOs.run(stage='training', scale_type='USO_scale', playback_direction='toward', sound_id='random')

USOs.run(stage='test', n_reps=4, scale_type='USO_scale', record_response=True, sound_id=18)

USOs.play_deviant()

USOs.run(stage='experiment', scale_type='USO_scale', n_reps=55, isi=1.5)
