import pathlib

DIR = pathlib.Path(__file__).parent.absolute()


def get_config():
    config_obj = {}
    proc_list = [['RP2', 'RP2', DIR / 'data' / 'bi_play_buf.rcx'],
                 ['RX81', 'RX8', DIR / 'data' / 'play_buf.rcx'],
                 ['RX82', 'RX8', DIR / 'data' / 'play_buf.rcx']]
    distances = {
        'detailed': [20, 40, 80, 160, 320, 640, 960, 1280, 1920, 2560],
        'sparse': [20, 160, 640, 1280, 2560]
    }
    config_obj['proc_list'] = proc_list
    config_obj['distances'] = distances
    return config_obj
