import pathlib
import numpy

DIR = pathlib.Path(__file__).parent.absolute()


def get_config():
    config_obj = {}
    proc_list = [['RP2', 'RP2', DIR / 'data' / 'bi_play_buf.rcx'],
                 ['RX81', 'RX8', DIR / 'data' / 'play_buf.rcx'],
                 ['RX82', 'RX8', DIR / 'data' / 'play_buf.rcx']]
    distance_groups = {
        'linear_10': {
            1: [20],
            2: [40],
            3: [80],
            4: [160],
            5: [320],
            6: [640],
            7: [960],
            8: [1280],
            9: [1920]
        },
        'linear_5': {
            1: numpy.arange(20, 80, 20),
            2: numpy.arange(500, 580, 20),
            3: numpy.arange(980, 1060, 20),
            4: numpy.arange(1460, 1540, 20),
            5: numpy.arange(1940, 2020, 20)
        },
        'log_10': {
            1: [20, 40],
            2: [100, 120],
            3: [400, 420],
            4: [800, 820, 840],
            5: [1000, 1020],
            6: [1200, 1220],
            7: [1500, 1520],
            8: [1760, 1780, 1800],
            9: [2000, 2020, 2040],
            10: [2460, 2480, 2500]
        },
        'log_5': {
            1: numpy.arange(20, 80, 20),
            2: numpy.arange(380, 440, 20),
            3: numpy.arange(780, 840, 20),
            4: numpy.arange(1180, 1240, 20),
            5: numpy.arange(1580, 1580, 20)
        },
        'log_5_full': {
            1: numpy.arange(20, 60, 20),
            2: numpy.arange(200, 280, 20),
            3: numpy.arange(780, 860, 20),
            4: numpy.arange(1360, 1440, 20),
            5: numpy.arange(1940, 2020, 20)
        },
        'vocal_effort': {
            '1': ['dis1'],
            '2': ['dis2'],
            '3': ['dis3'],
            '4': ['dis4'],
            '5': ['dis5']
        }
    }
    config_obj['proc_list'] = proc_list
    config_obj['distance_groups'] = distance_groups
    return config_obj
