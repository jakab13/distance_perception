from experiment.trials import Trials, Experiment

experiment = Experiment()
experiment.initialise()

participant_id = 'test'

vocalist = Trials(sound_type="vocalist-11-short", participant_id=participant_id)

vocalist.run(stage='training', playback_direction='away')
vocalist.run(stage='training', playback_direction='toward')

vocalist.run(stage='test', n_reps=4, record_response=True)

vocalist.play_deviant()

vocalist.run(stage='experiment', n_reps=60)
