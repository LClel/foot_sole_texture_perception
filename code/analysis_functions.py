""" Analysis functions used in analysis_pipeline.py
Author: Luke Cleland, ORCID: 0000-0001-8486-2780. GitHub: LClel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats, spatial
from constants import *
from scipy.stats import f_oneway
import math
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch
from statsmodels.multivariate.manova import MANOVA
from corrstats import *
from scipy.optimize import curve_fit
from sklearn import linear_model
import statsmodels.api as sm

def participant_demographics():
    """ Print the mean and standard deviation age across participants, and print male-female split

    :return:
    """

    mean_age = np.mean(list(participant_age.values()))
    print('Mean participant age:', mean_age)
    std_age = np.std(list(participant_age.values()))
    print('SD participant age:', std_age)

    n_male = list(participant_sex.values()).count('male')
    print('Number of males:', n_male)
    n_female = list(participant_sex.values()).count('female')
    print('Number of females:', n_female)


def import_data(participant_id):
    """ Load in participant data

    :param participant_id:
    :return:
    """

    # load in raw data file
    data = pd.ExcelFile('../data/' + participant_id + '/' + participant_id + '_responses.xlsx')

    # loop through conditions
    for sheet in ['walking', 'sitting', 'hand']:

        if sheet == 'walking':

            # start dataframe
            df = pd.read_excel(data, 'walking')

        else:
            # add other sheets to dataframe
            df = df.append(pd.read_excel(data, sheet))
            df.drop(df.filter(regex="Unname"), axis=1, inplace=True)

    # assign participant ID to data
    df['Participant'] = participant_id

    df = df.reset_index(drop=True)

    # add texture name to each texture number
    for i in range(1,17):

        idxs = df[df['Texture'] == i].index
        # print(idxs)

        df.loc[idxs, 'Name'] = texture_names_simple[i]

    return df


def ratio_relative_to_mean(df):
    """ Calculate mean rating of each metric for a participant.
        Divide all ratings for that metric by the mean to create ratio

    :return:
        normalized df
    """

    # create columns to store data
    df['Ratio'] = np.zeros(df.shape[0])
    df['Mean ratio'] = np.zeros(df.shape[0])
    #print(df)
    #exit()

    # loop through conditions
    for condition in df['Condition'].unique():

        # loop through metrics
        for metric in df['Metric'].unique():

            # select relevant combination-metric combination
            df_metric = df[(df['Condition'] == condition) & (df['Metric'] == metric)]

            # add 0.1 so that 0 values are normalized properly
            df_metric['Rating'] = df_metric['Rating'] + 0.1

            # calculate mean rating across 3 trials
            mean_rating = df_metric['Rating'].mean(skipna=True)

            # get the indexes for relevant rows
            idxs = df[(df['Condition'] == condition) & (df['Metric'] == metric)].index

            # transform ratings to ratio by dividing by mean rating
            df.loc[idxs, 'Ratio'] = df_metric['Rating'] / mean_rating

            # loop through texture numbers
            for i in range(1, 17):

                # extract relevant condition-metric-texture combinations
                df_texture = df[(df['Condition'] == condition) & (df['Metric'] == metric) & (df['Texture'] == i)]

                # calculate mean ratio
                mean_ratio = df_texture['Ratio'].mean()

                # get indexes relevant condition-metric-texture combination
                idxs = df[(df['Condition'] == condition) & (df['Metric'] == metric) & (df['Texture'] == i)].index

                # assign mean ratio for each condition-metric-texture combination
                df.loc[idxs, 'Mean ratio'] = mean_ratio

    return df


def rank_tetures(df):
    """ Add rank to each texture based on mean ratio

    :param df:
    :return:
    """

    # create column to store rank
    df['Rank'] = np.zeros(df.shape[0])

    # loop through conditions
    for condition in df['Condition'].unique():

        # loop through metrics
        for metric in df['Metric'].unique():

            # extract condition-metric combination
            df_metric = df[(df['Condition'] == condition) & (df['Metric'] == metric)]

            # group by texture and calcualte mean ratio
            grouped = df_metric.groupby('Texture').mean()

            # order ratio to find rank
            rank = stats.rankdata(grouped['Ratio'])

            # loop through ranks
            for i in range(len(rank)):

                # extract condition-metric-texture combination
                idxs = df[(df['Condition'] == condition) & (df['Metric'] == metric) & (df['Texture'] == i+1)].index

                # assign rank to column
                df.loc[idxs, 'Rank'] = rank[i]

    return df


def standard_error_responses(df):
    """ Calculate standard error of the rating of each metric for a participant.

    :return:
        normalized df
    """

    # create column to store standard error
    df['SE'] = np.zeros(df.shape[0])

    # loop through conditions
    for condition in conditions:

        # loop through metrics
        for metric in df['Metric'].unique():

            # loop through textures
            for texture in range(1,17):

                # extract condition-metric-texture combination
                df_metric = df[(df['Condition'] == condition) & (df['Metric'] == metric) & (df['Texture'] == texture)]

                # extract ratings
                ratings = df_metric['Rating'].values

                # calculate standard error
                SE_texture_rating = stats.sem(ratings, nan_policy='omit')

                # find row indexes for condition-metric-texture combination
                idxs = df[(df['Condition'] == condition) & (df['Metric'] == metric) & (df['Texture'] == texture)].index

                # assign standard error value
                df.loc[idxs, 'SE'] = SE_texture_rating

    return df


def collate_all_data_ratio(participant_ids):
    """ Collate data from all participants into one dataframe

    :param participant_ids:
    :return:
    """

    # loop through participant ids
    for i in range(len(participant_ids)):


        if i == 0:
            df = import_data(participant_ids[i]) # import data
            df = ratio_relative_to_mean(df) # transform ratings into ratio
            df = rank_tetures(df) # calcualte rank of each texture
            df = standard_error_responses(df) # calculate standard error of responses

        else:
            ppt_df = import_data(participant_ids[i]) # import data
            ppt_df = ratio_relative_to_mean(ppt_df) # transform ratings into ratio
            ppt_df = rank_tetures(ppt_df) # calcualte rank of each texture
            ppt_df = standard_error_responses(ppt_df) # calculate standard error of responses

            df = df.append(ppt_df) # join dataframes

    # select only relevant columns
    df = df[['Texture', 'Condition', 'Trial', 'Metric', 'Rating',
       'Participant', 'Name', 'Ratio', 'Mean ratio', 'Rank', 'SE']]

    return df


def average_intersubject_correlation(df):
    """ Calcualte the correlation between all participants, and calculate average correlation

    :param df:
    :return:
    """

    # select only trial 2 so one value per texture per participant-condition-metric combination
    df = df[df['Trial'] == 2.]

    # create new dataframe to store results
    mean_r_df = pd.DataFrame({'Comparison': [], 'mean r': [], 'min r': [], 'max r': [], 'std r': []})

    # loop through conditions
    for condition in conditions:

        # loop through metrics
        for metric in metrics[:-1]:

            # select condition-metric combination
            trimmed_df = df[(df['Condition'] == condition) & (df['Metric'] == metric)]

            # remove participant 12 hand condition
            if condition == 'hand':
                trimmed_df = trimmed_df[trimmed_df['Participant'] != 'PPT_012']

            # get participant IDs
            participants = trimmed_df['Participant'].unique()

            n_comparisons = 0
            correlations = np.array([])
            # loop through participants
            for i in range(len(participants)-1):

                for j in range(len(participants[i+1:])):

                    n_comparisons += 1

                    # get mean ration for each participant in correlation
                    ppt_i = trimmed_df[trimmed_df['Participant'] == participants[i]]['Mean ratio'].values
                    ppt_j = trimmed_df[trimmed_df['Participant'] == participants[i+1:][j]]['Mean ratio'].values

                    # run spearmans rank correlation
                    correlate = stats.spearmanr(ppt_i, ppt_j)

                    # add correlation to array
                    correlations = np.append(correlations, correlate[0])

            # calculate mean, min and max correlation
            # store data in dataframe
            correlation_metric_condition = pd.DataFrame({'Comparison': [condition + ': ' + metric], \
                                                         'mean r': np.mean(correlations),
                                                         'min r': np.min(correlations), 'max r': np.max(correlations), \
                                                         'std r': np.std(correlations)})

            # join dataframes
            mean_r_df = mean_r_df.append(correlation_metric_condition)

    # save as .csv file
    mean_r_df.to_csv('../stats_output/inter_subject_correlations.csv')

    return mean_r_df