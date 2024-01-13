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


def correlate_metrics_between_conditions(df):
    """ Run a correlation for each metric between conditions

    :param df:
    :return:
    """

    df = df[df['Trial'] == 2.0] # only use mean ratio per participant

    # set up plot
    fig = plt.figure(constrained_layout=True, dpi=300, figsize=(8, 2))
    plt.rcParams.update({'font.size': 5})
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    axes = [ax1, ax2, ax3]

    i = 0
    # loop through metrics
    for metric in dimensions['sitting']:

        # extract data for metric
        metric_df = df[df['Metric'] == metric]

        # generate pivot table for mean ratio on the metric for all conditions
        pivoted = pd.pivot_table(metric_df, values='Mean ratio', index=['Texture'],
                                 columns=['Condition'])

        # plot correlation heatmap
        sns.heatmap(pivoted.corr(method='spearman'), vmin=0., vmax=1., ax=axes[i], square=True, annot=True, fmt=".2f", )
        axes[i].set_title(metric)

        i += 1

    # save figure
    plt.savefig('../figures/single_value_per_texture_correlating_metrics_between_conditions.png')

    # ------------------------ get p-values ------------------------ #
    # create dataframe to store results
    correlation_df = pd.DataFrame({'Metric': [], 'covariate1': [], 'covariate2': [], 'r': [], 'p': []})

    # loop through metrics
    for metric in dimensions['sitting']:

        # extract data for given metric
        metric_df = df[df['Metric'] == metric]

        # loop through conditions
        for condition1 in dimensions:

            # extract data for condition
            condition_df_1 = metric_df[metric_df['Condition'] == condition1]

            # loop through conditions
            for condition2 in dimensions:

                # extract data for condition
                condition_df_2 = metric_df[metric_df['Condition'] == condition2]

                # remove participant 12 from any comparison involving the hand condition, which they did not complete
                if condition1 == 'hand' or condition2 == 'hand':
                    condition_df_1 = condition_df_1[condition_df_1['Participant'] != 'PPT_012']
                    condition_df_2 = condition_df_2[condition_df_2['Participant'] != 'PPT_012']

                # calculate mean score for texture across all participants
                condition_df_1 = condition_df_1.sort_values(by=['Texture']).groupby('Texture').mean()

                # calculate mean score for texture across all participants
                condition_df_2 = condition_df_2.sort_values(by=['Texture']).groupby('Texture').mean()

                # calculate spearmans rank correlation
                corr = stats.spearmanr(condition_df_1['Mean ratio'], condition_df_2['Mean ratio'])

                # store data into dataframe
                df_single = pd.DataFrame(
                    {'Metric': [metric], 'covariate1': [condition1], 'covariate2': [condition2], 'r': [corr[0]],
                     'p': [corr[1]]})

                # join dataframes
                correlation_df = pd.concat([correlation_df, df_single])

    # save results to .csv file
    correlation_df.to_csv('../stats_output/single_value_per_texture_correlating_metrics_between_conditions.csv')


def inter_participant_correlate_metrics_between_conditions(df):
    """ Run a correlation for each metric between conditions for each participant

    :param df:
    :return:
    """

    df = df[df['Trial'] == 2.0] # only use mean ratio per participant

    # create dataframe to store results
    correlation_df = pd.DataFrame({'Metric': [], 'covariate1': [], 'covariate2': [], \
                                   'participant': [], 'r': [], 'p': []})

    # loop through metrics
    for metric in dimensions['sitting']:

        # extract data for the relevant metric
        metric_df = df[df['Metric'] == metric]

        # loop through participant ids
        for participant in participant_ids:

            # extract data for relevant participant
            ppt_df = metric_df[metric_df['Participant'] == participant]

            # loop through conditions
            for condition1 in dimensions:

                # extract data for relevant condition
                condition_df_1 = ppt_df[ppt_df['Condition'] == condition1]

                # loop through conditions
                for condition2 in dimensions:

                    # extract data for relevant condition
                    condition_df_2 = ppt_df[ppt_df['Condition'] == condition2]

                    # remove participant 12 from any comparison involving the hand condition, which they did not complete
                    if condition1 == 'hand' or condition2 == 'hand':
                        if participant == 'PPT_012':
                            continue
                        else:
                            # calculate spearmans rank correlation
                            corr = stats.spearmanr(condition_df_1['Mean ratio'], condition_df_2['Mean ratio'])

                    else:
                        # calculate spearmans rank correlation
                        corr = stats.spearmanr(condition_df_1['Mean ratio'], condition_df_2['Mean ratio'])

                    # store data in dataframe
                    df_single = pd.DataFrame(
                        {'Metric': [metric], 'covariate1': [condition1], 'covariate2': [condition2], \
                         'participant': [participant], 'r': [corr[0]],
                         'p': [corr[1]]})

                    # join dataframes
                    correlation_df = pd.concat([correlation_df, df_single])

    # save result as .csv file
    correlation_df.to_csv('../stats_output/ppt_single_value_per_texture_correlating_metrics_between_conditions2.csv')

def calculate_mean_ppt_level_correlations_metrics_between_conditions():
    """ Calculate mean correlation for each metric between conditions for each participant

    :return:
    """

    # load in participant level correlations
    df = pd.read_csv('../stats_output/ppt_single_value_per_texture_correlating_metrics_between_conditions2.csv')

    # generate dataframe to store data
    mean_corr_df = pd.DataFrame({'Metric': [], 'covariate1': [], 'covariate2': [], 'mean r': [], 'std r': []})

    # loop through metrics
    for metric in dimensions['sitting']:

        # extract data for relevant metric
        metric_df = df[df['Metric'] == metric]

        # loop through conditions
        for condition1 in dimensions:

            # loop through conditions
            for condition2 in dimensions:

                # extract relevant comparison
                comparison_df = metric_df[(metric_df['covariate1'] == condition1) & (metric_df['covariate2'] == condition2)]

                # calculate mean correlation
                # store data in dataframe
                df_single = pd.DataFrame(
                    {'Metric': [metric], 'covariate1': [condition1], 'covariate2': [condition2], \
                     'mean r': comparison_df['r'].mean(), 'std r': comparison_df['r'].std()})

                # join dataframes
                mean_corr_df = pd.concat([mean_corr_df, df_single])

    # save result as .csv file
    mean_corr_df.to_csv('../stats_output/mean_ppt_correlation_metrics_between_conditions.csv')


def non_parametric_rm_ANOVA(df):
    """ Run a non-parametric repeated measures ANOVA - Friedman's test
    To compare the mean ranks of each texture for a given metric across conditions

    :param df:
    :param metric:
    :return:
    """

    # use only trial 2 so one value per condition-metric-texture-participant combination
    df = df[df['Trial'] == 2.0]

    # create dataframe to store result
    friedman_df = pd.DataFrame({'Metric': [], 'Texture number': [], 'Texture name': [], 'F': [], 'p': []})

    # remove participant 12 from analysis as did not complete the hand condition
    df = df[df['Participant'] != 'PPT_012']

    # loop through metrics (not including stability)
    for metric in metrics[:-1]:

        # extract data for relevant metric
        metric_df = df[df['Metric'] == metric]

        # remove metric column
        metric_df.drop(columns=['Metric'])

        # create pivot table to rearrange data
        pivoted = pd.pivot_table(metric_df, values='Mean ratio', index=['Texture', 'Participant'],
                                 columns=['Condition'])

        # create dataframe for the given metric
        friedman_df_metric = pd.DataFrame({'Metric': [], 'Texture': [], 'F': [], 'p': []})

        # loop through textures
        for i in range(1,17):

            # extract all values for the texture in each of the conditions (1 per participant)
            hand = pivoted.loc[i]['hand'].values
            sitting = pivoted.loc[i]['sitting'].values
            walking = pivoted.loc[i]['walking'].values

            # run friedman analysis
            friedman = stats.friedmanchisquare(hand, sitting, walking)

            # store data in dataframe
            texture_df = pd.DataFrame({'Metric': [metric], 'Texture number': [i], 'Texture name': [texture_names_simple[i]], 'F': [friedman[0]], 'p': [friedman[1]]})

            # join dataframes
            friedman_df_metric = pd.concat([friedman_df_metric, texture_df], ignore_index=True)

        # join dataframes
        friedman_df = pd.concat([friedman_df, friedman_df_metric], ignore_index=True)

    # save result as .csv file
    friedman_df.to_csv('../stats_output/friedman.csv')



def non_parametric_rm_t_test(df):
    """ Run a non-parametric repeated measures t-test - Wilcoxen signed rank test
    To compare the mean ranks of each texture for a given metric across conditions

    :param df:
    :param metric:
    :return:
    """

    # use only
    df = df[df['Trial'] == 2.]

    # create dataframe to store result
    wilcoxen_df = pd.DataFrame({'Metric': [], 'Texture number': [], 'Texture name': [], 'Comparison': [], 'W': [], 'p': []})

    # remove participant 12 from analysis as did not complete the hand condition
    df = df[df['Participant'] != 'PPT_012']

    # loop through metrics except stability
    for metric in metrics[:-1]:

        # extract data for relevant metric
        metric_df = df[df['Metric'] == metric]

        # remove metric column
        metric_df.drop(columns=['Metric'])

        # create pivot table to rearrange data
        pivoted = pd.pivot_table(metric_df, values='Mean ratio', index=['Texture', 'Participant'],
                                 columns=['Condition'])

        # store data in dataframe for metric
        wilcoxen_df_metric = pd.DataFrame({'Metric': [], 'Texture': [], 'Comparison': [], 'W': [], 'p': []})

        # loop through textures
        for i in range(1,17):

            # extract all values for the texture in each of the conditions (1 per participant)
            hand = pivoted.loc[i]['hand'].values
            sitting = pivoted.loc[i]['sitting'].values
            walking = pivoted.loc[i]['walking'].values

            # run wilcoxen tests on each condition comparison
            hand_sitting = stats.wilcoxon(hand, sitting)
            hand_walking = stats.wilcoxon(hand, walking)
            sitting_walking = stats.wilcoxon(sitting, walking)

            # save result for each comparison
            h_s_df = pd.DataFrame({'Metric': [metric], 'Texture number': [i], 'Texture name': [texture_names_simple[i]], \
                                   'Comparison': 'hand-sitting', 'W': [hand_sitting[0]], 'p': [hand_sitting[1]]})
            h_w_df = pd.DataFrame({'Metric': [metric], 'Texture number': [i], 'Texture name': [texture_names_simple[i]], \
                                   'Comparison': 'hand-walking', 'W': [hand_walking[0]], 'p': [hand_walking[1]]})
            s_w_df = pd.DataFrame({'Metric': [metric], 'Texture number': [i], 'Texture name': [texture_names_simple[i]], \
                                   'Comparison': 'sitting-walking', 'W': [sitting_walking[0]], 'p': [sitting_walking[1]]})

            # join dataframes together
            wilcoxen_df_metric = pd.concat([wilcoxen_df_metric, h_s_df, h_w_df, s_w_df], ignore_index=True)

        # store data in dataframe
        wilcoxen_df = pd.concat([wilcoxen_df, wilcoxen_df_metric], ignore_index=True)

    # save result as .csv file
    wilcoxen_df.to_csv('../stats_output/wilcoxen.csv')


def mean_ratio_textures(df):
    """ Plot the mean ratio scores per texture, including responses from each participant
    Significant Friedman's are indicated on plot

    :param df:
    :return:
    """

    # ensure only mean ratio is used per participant, condition, metric
    df = df[df['Trial'] == 2]

    # close all other plots
    plt.close('all')

    # define plot structure
    fig = plt.figure(constrained_layout=True, dpi=300, figsize=(12, 12))
    plt.rcParams.update({'font.size': 10})
    gs = GridSpec(3, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # load in Friedman analysis
    friedman = pd.read_csv('../stats_output/friedman.csv')

    # remove participant 12 hand condition
    df = df.loc[~((df['Condition'] == 'hand') & (df['Participant'] == 'PPT_012'))]

    # set color palette
    sns.set_palette(condition_colors)

    # ------------------------------------------------ Roughness ------------------------------------------------ #

    # extract roughness data
    df_roughness = df[df['Metric'] == 'roughness']

    # sort data from most rough to least rough and save, then reload
    df_roughness.groupby('Name').mean().sort_values(by='Mean ratio', ascending=False)['Mean ratio'].to_csv(
        '../processed_data/roughness_rank.csv')
    sorted = pd.read_csv('../processed_data/roughness_rank.csv')['Name']

    # plot mean rating for each participant
    sns.stripplot(data=df_roughness, x="Name", y="Mean ratio", hue="Condition",
                  hue_order=['walking', 'sitting', 'hand'], dodge=True, alpha=.1, ax=ax1, order=sorted)
    # plot mean rating from all participants
    sns.pointplot(
        data=df_roughness, x="Name", y="Mean ratio", hue="Condition",
        hue_order=['walking', 'sitting', 'hand'], marker="-", ax=ax1, join=False,
        dodge=.5, markers="_", markersize=20, errorbar=0,
        errwidth=0, order=sorted, legend=False)
    ax1.legend([], [], frameon=False)
    ax1.set_title('Smooth - Rough')
    ax1.set_xticklabels(sorted, rotation=80)
    ax1.set_ylabel('Mean ratio')
    ax1.set_xlabel('Texture')
    ax1.set_ylim(0, 4.5)
    sns.despine(ax=ax1)

    # identify significant Friedman's test and indicate on plot
    for i in range(len(sorted)):
        p = friedman[(friedman['Metric'] == 'roughness') & (friedman['Texture name'] == sorted[i])]
        p = np.array(p['p'])
        if p < .05:
            ax1.scatter(i, 4.25, marker='*', c='black')

    # ------------------------------------------------ Hardness ------------------------------------------------ #

    # extract hardness data
    df_hardness = df[df['Metric'] == 'hardness']

    # plot mean rating for each participant
    sns.stripplot(data=df_hardness, x="Name", y="Mean ratio", hue="Condition",
                  hue_order=['walking', 'sitting', 'hand'], dodge=True, alpha=.1, ax=ax2, order=sorted)
    # plot mean rating from all participants
    sns.pointplot(
        data=df_hardness, x="Name", y="Mean ratio", hue="Condition",
        hue_order=['walking', 'sitting', 'hand'], marker="-", ax=ax2, join=False,
        dodge=.5, markers="_", markersize=20, errorbar=0,
        errwidth=0, order=sorted, legend=False)
    ax2.legend([], [], frameon=False)
    ax2.set_title('Soft - Hard')
    ax2.set_xticklabels(sorted, rotation=80)
    ax2.set_ylabel('Mean ratio')
    ax2.set_xlabel('Texture')
    ax2.set_ylim(0, 4.5)
    sns.despine(ax=ax2)

    # identify significant Friedman's test and indicate on plot
    for i in range(len(sorted)):
        p = friedman[(friedman['Metric'] == 'hardness') & (friedman['Texture name'] == sorted[i])]
        p = np.array(p['p'])
        if p < .05:
            ax2.scatter(i, 4.25, marker='*', c='black')


    # ------------------------------------------------ Slipperiness ------------------------------------------------ #

    # extract data for slipperiness
    df_slipperiness = df[df['Metric'] == 'slipperiness']

    # plot mean rating for each participant
    sns.stripplot(data=df_slipperiness, x="Name", y="Mean ratio", hue="Condition",
        hue_order=['walking','sitting','hand'], dodge=True, alpha=.1, ax=ax3, order=sorted)
    # plot mean rating from all participants
    sns.pointplot(
        data=df_slipperiness, x="Name", y="Mean ratio", hue="Condition",
        hue_order=['walking','sitting','hand'], marker="-", ax=ax3, join=False,
        dodge=.55, markers="_", markersize=40, errorbar=0, errwidth=0, order=sorted, legend=False)
    ax3.legend([], [], frameon=False)
    ax3.set_title('Slippery - Sticky')
    ax3.set_xticklabels(sorted, rotation=80)
    ax3.set_ylabel('Mean ratio')
    ax3.set_xlabel('Texture')
    ax3.set_ylim(0, 4.5)
    sns.despine(ax=ax3)

    # identify significant Friedman's test and indicate on plot
    for i in range(len(sorted)):
        p = friedman[(friedman['Metric'] == 'slipperiness') & (friedman['Texture name'] == sorted[i])]
        p = np.array(p['p'])
        if p < .05:
            ax3.scatter(i, 4.25, marker='*', c='black')

    # space subplots
    plt.tight_layout()
    # save figure
    plt.savefig('../figures/mean_ratio_per_texture.png')