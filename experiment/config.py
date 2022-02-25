import pathlib
import os

# DIR = pathlib.Path(os.getcwd()) / 'experiment'
DIR = pathlib.Path(os.getcwd())

def get_config():
    config_obj = {}
    proc_list = [['RP2', 'RP2', DIR / 'data' / 'bi_play_buf.rcx'],
                 ['RX81', 'RX8', DIR / 'data' / 'play_buf.rcx'],
                 ['RX82', 'RX8', DIR / 'data' / 'play_buf.rcx']]
    distances = {
        'detailed': [0.2, 0.4, 0.6, 0.8, 1, 1.6, 2.2, 3.0, 4, 5, 7, 10, 13, 16, 18],
        'sparse': [0.2, 3, 10, 18]
    }
    config_obj['proc_list'] = proc_list
    config_obj['distances'] = distances
    return config_obj