import pathlib
import os

DIR = pathlib.Path(__file__).parent.absolute()

def get_config():
    config_obj = {}
    proc_list = [['RP2', 'RP2', DIR / 'data' / 'bi_play_buf.rcx'],
                 ['RX81', 'RX8', DIR / 'data' / 'play_buf.rcx'],
                 ['RX82', 'RX8', DIR / 'data' / 'play_buf.rcx']]
    distances = {
        'detailed': [0.2, 0.6, 1, 2, 3, 5, 7, 9, 12, 15, 18],
        'sparse': [0.2, 2, 10, 18]
    }
    config_obj['proc_list'] = proc_list
    config_obj['distances'] = distances
    return config_obj