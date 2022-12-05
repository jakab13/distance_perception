from experiment.trials import Trials, Experiment

experiment = Experiment()
experiment.initialise()

participant_id = 'test'

noise = Trials(sound_type="pinknoise_ramped", participant_id=participant_id)

noise.run(stage='training', scale_type='log_5', playback_direction='away')
noise.run(stage='training', scale_type='log_5', playback_direction='toward')
noise.run(stage='training', scale_type='log_10', playback_direction='away')
noise.run(stage='training', scale_type='log_10', playback_direction='toward')

noise.run(stage='test', n_reps=4, record_response=True)

noise.run_control(n_reps=60)

noise.play_deviant()

noise.run(stage='experiment', scale_type='log_5', n_reps=60)
