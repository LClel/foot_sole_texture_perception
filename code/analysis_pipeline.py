""" Analysis pipeline to generate all stats and figures in the associated paper
Author: Luke Cleland, ORCID: 0000-0001-8486-2780. GitHub: LClel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from analysis_functions import *

# calculate and print participant mean (std) age
participant_demographics()

# import  raw data and collate into one dataframe
#df = collate_all_data_ratio(['PPT_001','PPT_002','PPT_003', 'PPT_004', 'PPT_005', 'PPT_006', 'PPT_007',\
#                             'PPT_008','PPT_009','PPT_010','PPT_011', 'PPT_012', 'PPT_013','PPT_014','PPT_015',\
#                             'PPT_016', 'PPT_017','PPT_018','PPT_019','PPT_020'])

# save dataframe with all participant data as a .csv
#df.to_csv('../processed_data/collated_data.csv')

# load in dataframe with all participant data
df = pd.read_csv('../processed_data/collated_data.csv')


# calculate difference between scores in each condition-metric combination
stats_scores_over_trials(df)

# calculate average intersubject correlation for each condition-metric combination
average_intersubject_correlation(df)

# calculate correlations for each metric across conditions
correlate_metrics_between_conditions(df)

# compare spread of scores between conditions
spread_of_scores_between_conditions(df)

# compare between and within participant correlations
correlation_comparison_stats()

# plot rank per texture across conditions
tidy_rank_per_condition(df)

# calculate participant level correlations for each metric across conditions
inter_participant_correlate_metrics_between_conditions(df)

# calculate mean correlations for each metric across conditions at participant level
calculate_mean_ppt_level_correlations_metrics_between_conditions()

# run correlations between conditions at texture level
texture_level_correlate_metrics_between_conditions_textures(df)

# compare correlations for significance
correlation_comparison()

# run Friedman non parametric repeated measures ANOVA and save result to .csv
non_parametric_rm_ANOVA(df)

# run Wilcoxen post-hoc non parametric repeated measures t-test and save result to .csv
non_parametric_rm_t_test(df)

# run Friedman non parametric repeated measures ANOVA and save result to .csv
non_parametric_rm_ANOVA_rank(df)

# run Wilcoxen post-hoc non parametric repeated measures t-test and save result to .csv
non_parametric_rm_t_test_rank(df)

# plot mean rank across all conditions with significance indicators
mean_rank_textures(df)

# plot mean rating across all conditions with significance indicators
mean_ratio_textures(df)

# print minimum and maximum mean ration per condition-metric combination
mean_ratio_spread(df)

# calculate correlations for each metric within conditions
correlate_metrics_within_conditions(df)

# calculate participant level correlations for each metric within conditions
inter_participant_correlate_metrics_within_conditions(df)

# calculate mean correlations for each metric within conditions at participant level
calculate_mean_ppt_level_correlations_metrics_within_conditions()

# calculate multiple regression on stability
# plot scatterplots for each metric and it's relation to stability
multiple_regression(df)