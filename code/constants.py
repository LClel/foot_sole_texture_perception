import numpy as np

# participant ID
participant_ids = ['PPT_001','PPT_002','PPT_003','PPT_004','PPT_005','PPT_006','PPT_007','PPT_008','PPT_009','PPT_010',\
                   'PPT_011','PPT_012','PPT_013','PPT_014','PPT_015','PPT_016','PPT_017','PPT_018','PPT_019','PPT_020']

# participant age (years)
participant_age = {'PPT_001': 20,
                   'PPT_002': 21,
                   'PPT_003': 22,
                   'PPT_004': 29,
                   'PPT_005': 18,
                   'PPT_006': 18,
                   'PPT_007': 18,
                   'PPT_008': 19,
                   'PPT_009': 18,
                   'PPT_010': 19,
                   'PPT_011': 19,
                   'PPT_012': 18,
                   'PPT_013': 24,
                   'PPT_014': 18,
                   'PPT_015': 19,
                   'PPT_016': 19,
                   'PPT_017': 20,
                   'PPT_018': 23,
                   'PPT_019': 19,
                   'PPT_020': 19}

# participant sex (male/female)
participant_sex = {'PPT_001': 'male',
                   'PPT_002': 'male',
                   'PPT_003': 'male',
                   'PPT_004': 'male',
                   'PPT_005': 'female',
                   'PPT_006': 'female',
                   'PPT_007': 'female',
                   'PPT_008': 'female',
                   'PPT_009': 'female',
                   'PPT_010': 'male',
                   'PPT_011': 'male',
                   'PPT_012': 'female',
                   'PPT_013': 'female',
                   'PPT_014': 'female',
                   'PPT_015': 'female',
                   'PPT_016': 'female',
                   'PPT_017': 'female',
                   'PPT_018': 'male',
                   'PPT_019': 'female',
                   'PPT_020': 'female'}

# names of perceptual dimensions
metrics = ['roughness','hardness','slipperiness','stability']

# names of presentation conditions
conditions = ['walking', 'sitting', 'hand']

# perceptual dimensions investigated per presentation condition
dimensions = {'walking': ['roughness','hardness','slipperiness','stability'],
              'sitting': ['roughness','hardness','slipperiness'],
              'hand': ['roughness','hardness','slipperiness']}

# layout of textures for stepping condition
layout = np.array([[3, 5, 1, 14],
                   [2, 11, 8, 13],
                   [6, 10, 16, 9],
                   [7, 15, 4, 12]])

# order of texture presentation for stepping condition
stepping_order = {'roughness': {1: np.array([[4, 5, 12, 13],
                                              [3, 6, 11, 14],
                                              [2, 7, 10, 15],
                                              [1, 8, 9, 16]]),
                                2: np.array([[12, 11, 10, 9],
                                              [13, 14, 7, 8],
                                              [16, 15, 6, 5],
                                              [1, 2, 3, 4]]),
                                3: np.array([[4, 3, 2, 1],
                                              [5, 6, 7, 8],
                                              [12, 11, 10, 9],
                                              [13, 14, 15, 16]])},
                  'hardness': {1: np.array([[16, 15, 14, 13],
                                              [9, 10, 11, 12],
                                              [8, 7, 6, 5],
                                              [1, 2, 3, 4]]),
                               2: np.array([[1, 8, 9, 16],
                                              [2, 7, 10, 15],
                                              [3, 6, 11, 14],
                                              [4, 5, 12, 13]]),
                               3: np.array([[13, 14, 15, 16],
                                              [12, 11, 10, 9],
                                              [5, 6, 7, 8],
                                              [4, 3, 2, 1]])},
                  'slipperiness': {1: np.array([[1, 2, 3, 4],
                                              [8, 7, 6, 5],
                                              [9, 10, 11, 12],
                                              [16, 15, 14, 13]]),
                                   2: np.array([[5, 6, 7, 8],
                                              [4, 3, 2, 1],
                                              [12, 11, 10, 9],
                                              [13, 14, 15, 16]]),
                                   3: np.array([[4, 3, 2, 1],
                                              [5, 14, 15, 16],
                                              [6, 13, 12, 11],
                                              [7, 8, 9, 10]])},
                  'stability': {1: np.array([[9, 8, 7, 6],
                                              [10, 11, 12, 5],
                                              [15, 14, 13, 4],
                                              [16, 1, 2, 3]]),
                                2: np.array([[13, 14, 15, 16],
                                              [4, 3, 2, 1],
                                              [12, 11, 10, 9],
                                              [5, 6, 7, 8]]),
                                3: np.array([[4, 3, 2, 1],
                                              [5, 6, 7, 10],
                                              [15, 14, 8, 9],
                                              [16, 13, 12, 11]])}}

# official name of each texture
textures = {1: 'TRAMPA door mat',
            2: 'TOFLUND rug',
            3: 'SALVIKEN hand towel',
            4: 'SVINDIGE rug',
            5: 'SUSIG desk pad',
            6: 'SLIRA place mat',
            7: 'STAGGSTARR chair pad',
            8: 'DOPPA bathtub mat',
            9: 'Astro turf',
            10: 'PLOJA place mat',
            11: 'Acacia garden deck tile',
            12: 'Tile',
            13: 'Crash mat',
            14: 'Firm foam pad',
            15: 'Gel pad',
            16: 'EVALI throw'}

# short names per texture
texture_short_name = {1: 'door mat',
                      2: 'rug 1',
                      3: 'towel',
                      4: 'rug 2',
                      5: 'cork mat',
                      6: 'plastic mat',
                      7: 'chair pad',
                      8: 'bath mat',
                      9: 'astro turf',
                      10: 'plastic desk mat',
                      11: 'garden decking',
                      12: 'tile',
                      13: 'crash mat',
                      14: 'foam pad',
                      15: 'gel pad',
                      16: 'throw'}

# simple names per texture
texture_names_simple = {1: 'Door mat',
                        2: 'Rug 1',
                        3: 'Towel',
                        4: 'Rug 2',
                        5: 'Cork pad',
                        6: 'Plastic place mat',
                        7: 'Chair pad',
                        8: 'Bath mat',
                        9: 'Astro turf',
                        10: 'Plastic desk mat',
                        11: 'Garden deck tile',
                        12: 'Tile',
                        13: 'Crash mat',
                        14: 'Firm foam pad',
                        15: 'Gel pad',
                        16: 'Throw'}

# -------- colors for plotting -------- #

# color for each texture number
texture_colors = {1: 'darkgoldenrod',
                  2: 'lightcoral',
                  3: 'dimgrey',
                  4: 'orangered',
                  5: 'darkorange',
                  6: 'indigo',
                  7: 'deepskyblue',
                  8: 'mediumturquoise',
                  9: 'limegreen',
                  10: 'silver',
                  11: 'teal',
                  12: 'cornflowerblue',
                  13: 'dodgerblue',
                  14: 'blue',
                  15: 'violet',
                  16: 'crimson'}

# color for each presentation condition
condition_colors = ['#ff218c','#21b1ff','#f2d602']