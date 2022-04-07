from trials import Trials

training = {
    'bark': Trials(sound_type="bark"),
    'pinknoise': Trials(sound_type="pinknoise"),
    'bum': Trials(sound_type="bum"),
    'lock': Trials(sound_type="lock"),
    'plug': Trials(sound_type="plug")
}

training['bark'].play_control()
training['bark'].run(playback_direction='away')
training['bark'].run(playback_direction='away', isi=0.7)
training['bark'].run(playback_direction='toward')
training['bark'].run(playback_direction='toward', isi=1.0)
training['bark'].run(n_reps=50, record_response=True)

training['bum'].play_control()
training['bum'].run(playback_direction='away')
training['bum'].run(playback_direction='away', isi=0.7)
training['bum'].run(playback_direction='toward')
training['bum'].run(playback_direction='toward', isi=1.0)
training['bum'].run(n_reps=50, record_response=True)

training['lock'].play_control()
training['lock'].run(playback_direction='away')
training['lock'].run(playback_direction='away', isi=0.3)
training['lock'].run(playback_direction='toward')
training['lock'].run(playback_direction='toward', isi=1.0)
training['lock'].run(n_reps=50, record_response=True)

training['pinknoise'].play_control()
training['pinknoise'].run(playback_direction='away')
training['pinknoise'].run(playback_direction='away', isi=0.6)
training['pinknoise'].run(playback_direction='toward')
training['pinknoise'].run(playback_direction='toward', isi=1.0)
training['pinknoise'].run(n_reps=50, record_response=True)

training['plug'].run(playback_direction='toward', isi=0.5)
