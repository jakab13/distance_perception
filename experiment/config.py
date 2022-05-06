import pathlib

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
            1: [20],
            2: [80, 100],
            3: [600, 620, 640],
            4: [1200, 1220, 1240],
            5: [1940, 1960, 1980, 2000]
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
            '1': [20],
            '2': [40],
            '3': [80],
            '4': [160],
            '5': [320, 340, 360]
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
