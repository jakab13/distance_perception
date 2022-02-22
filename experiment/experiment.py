from training import Training

training = {
    'bark': Training(sound_type="bark"),
    'pinknoise': Training(sound_type="pinknoise")
}

training['bark'].run(playback_direction='away', isi=0.5)
training['bark'].run(playback_direction='toward')
training['bark'].run(n_reps=50, record_response=True)

training['pinknoise'].run(playback_direction='away')
training['pinknoise'].run(playback_direction='toward', isi=3.0)
training['pinknoise'].run(n_reps=10, record_response=True)

