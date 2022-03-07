from training import Training

participant_id = '53a243'

training = {
    'pinknoise': Training(sound_type="pinknoise", participant_id=participant_id),
    'bum': Training(sound_type="bum", participant_id=participant_id),
    'plug': Training(sound_type="plug", participant_id=participant_id),
    'waterdrop': Training(sound_type="waterdrop", participant_id=participant_id)
}

training['waterdrop'].play_control()
training['waterdrop'].run(playback_direction='away')
training['waterdrop'].run(playback_direction='away', isi=0.7)
training['waterdrop'].run(playback_direction='toward')
training['waterdrop'].run(playback_direction='toward', isi=1.0)
training['waterdrop'].run(n_reps=1, record_response=True)

training['bum'].play_control()
training['bum'].run(playback_direction='away', isi=2.0)
training['bum'].run(playback_direction='away', isi=0.7)
training['bum'].run(playback_direction='toward')
training['bum'].run(playback_direction='toward', isi=1.0)
training['bum'].run(n_reps=50, record_response=True)

training['plug'].play_control()
training['plug'].run(playback_direction='away')
training['plug'].run(playback_direction='away', isi=0.7)
training['plug'].run(playback_direction='toward')
training['plug'].run(playback_direction='toward', isi=1.0)
training['plug'].run(n_reps=1, record_response=True)

training['pinknoise'].play_deviant()
training['pinknoise'].play_control()

