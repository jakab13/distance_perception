import training

training_bark = training.Training(sound_type="bark")
training_pinknoise = training.Training()

training_bark.run(n_reps=2, isi=1.0)
# training_bark.run(playback_direction='toward')
training_pinknoise.run(n_reps=2, isi=1.0)