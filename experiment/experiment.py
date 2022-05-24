from experiment.trials import Trials, Experiment

experiment = Experiment()
experiment.initialise()
participant_id = experiment.participant_id

vocalist = Trials(sound_type="vocalist-11", participant_id=participant_id)

vocalist.run(stage='training', playback_direction='away')
vocalist.run(stage='training', playback_direction='toward')

vocalist.run(stage='test', n_reps=2, record_response=True)

vocalist.play_deviant()

vocalist.run(stage='experiment', n_reps=75)
