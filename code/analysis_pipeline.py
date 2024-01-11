import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from analysis_functions import *


participant_demographics()

#df = collate_all_data_ratio(['PPT_001','PPT_002','PPT_003', 'PPT_004', 'PPT_005', 'PPT_006', 'PPT_007',\
#                             'PPT_008','PPT_009','PPT_010','PPT_011', 'PPT_012', 'PPT_013','PPT_014','PPT_015',\
#                             'PPT_016', 'PPT_017','PPT_018','PPT_019','PPT_020'])

#df.to_csv('../processed_data/collated_data.csv')

df = pd.read_csv('../processed_data/collated_data.csv')