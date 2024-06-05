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
            #df = df.append(pd.read_excel(data, sheet))
            df = pd.concat([df, pd.read_excel(data, sheet)])
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


def rank_textures(df):
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
            df = rank_textures(df) # calcualte rank of each texture
            df = standard_error_responses(df) # calculate standard error of responses

        else:
            ppt_df = import_data(participant_ids[i]) # import data
            ppt_df = ratio_relative_to_mean(ppt_df) # transform ratings into ratio
            ppt_df = rank_textures(ppt_df) # calcualte rank of each texture
            ppt_df = standard_error_responses(ppt_df) # calculate standard error of responses

            #df = df.append(ppt_df) # join dataframes
            df = pd.concat([df, ppt_df])

    # select only relevant columns
    df = df[['Texture', 'Condition', 'Trial', 'Metric', 'Rating',
       'Participant', 'Name', 'Ratio', 'Mean ratio', 'Rank', 'SE']]

    return df


def average_intersubject_correlation(df):

    df = df[df['Trial'] == 2.]

    # participants = df['Participant'].unique()

    mean_r_df = pd.DataFrame({'Comparison': [], 'mean r': [], 'min r': [], 'max r': []})

    all_r_df = pd.DataFrame({'Metric': [], 'Condition': [], 'participant1': [], 'participant2': [], 'r': []})
    for condition in conditions:

        for metric in metrics[:-1]:

            trimmed_df = df[(df['Condition'] == condition) & (df['Metric'] == metric)]

            # remove participant 12 hand condition
            if condition == 'hand':
                trimmed_df = trimmed_df[trimmed_df['Participant'] != 'PPT_012']

            participants = trimmed_df['Participant'].unique()

            n_comparisons = 0
            correlations = np.array([])
            for i in range(len(participants)-1):
                print(participants[i])

                for j in range(len(participants[i+1:])):
                    print(participants[i+1:][j])

                    n_comparisons += 1

                    ppt_i = trimmed_df[trimmed_df['Participant'] == participants[i]]['Mean ratio'].values
                    ppt_j = trimmed_df[trimmed_df['Participant'] == participants[i+1:][j]]['Mean ratio'].values

                    correlate = stats.spearmanr(ppt_i, ppt_j)

                    correlations = np.append(correlations, correlate[0])

                    indiv_r_df = pd.DataFrame({'Metric': [metric], 'Condition': [condition], 'participant1': [participants[i]], 'participant2': [participants[i+1:][j]], 'r': [correlate[0]]})

                    #all_r_df = all_r_df.append(indiv_r_df)
                    all_r_df = pd.concat([all_r_df, indiv_r_df])
                print(n_comparisons)
            correlation_metric_condition = pd.DataFrame({'Comparison': [condition + ': ' + metric], 'mean r': np.mean(correlations),
                                                         'min r': np.min(correlations), 'max r': np.max(correlations)})

            #mean_r_df = mean_r_df.append(correlation_metric_condition)
            mean_r_df = pd.concat([mean_r_df, correlation_metric_condition])

    mean_r_df.to_csv('../stats_output/inter_subject_correlations.csv')
    all_r_df.to_csv('../stats_output/all_inter_subject_correlations.csv')

    return mean_r_df


def stats_scores_over_trials(df):

    df = df.sort_values(['Participant', 'Texture'])

    df_results = pd.DataFrame({'Condition': [], 'Metric': [], 'Texture': [], 'F': [], 'p': []})

    for condition in dimensions:

        for metric in dimensions[condition]:


            df_combination = df[(df['Condition'] == condition) & (df['Metric'] == metric)]

            if condition == 'hand':
                df_combination = df_combination[df_combination['Participant'] != 'PPT_012']

            for i in range(1,17):

                df_texture = df_combination[df_combination['Texture'] == i]

                if (condition == 'hand') and (metric == 'slipperiness') and (i==13):

                trial_1 = df_texture[df_texture['Trial'] == 1.]['Ratio']
                trial_2 = df_texture[df_texture['Trial'] == 2.]['Ratio']
                trial_3 = df_texture[df_texture['Trial'] == 3.]['Ratio']

                #print(stats.friedmanchisquare(trial_1, trial_2, trial_3))
                rm_anova = stats.f_oneway(trial_1, trial_2, trial_3)

                df_results = pd.concat([df_results, pd.DataFrame({'Condition': [condition], 'Metric': [metric], 'Texture': [i], \
                                           'F': [rm_anova[0]], 'p': [rm_anova[1]]})])




    df_results.to_csv('../stats_output/scores_across_trials.csv')


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
        sns.heatmap(pivoted.corr(method='spearman'), vmin=0., vmax=1., ax=axes[i], square=True, annot=True, fmt=".3f", )
        axes[i].set_title(metric)

        i += 1

    # save figure
    plt.savefig('../individual_figures/single_value_per_texture_correlating_metrics_between_conditions.png')

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
                    condition_df_1_corr = condition_df_1[condition_df_1['Participant'] != 'PPT_012']
                    condition_df_2_corr= condition_df_2[condition_df_2['Participant'] != 'PPT_012']

                else:
                    condition_df_1_corr = condition_df_1.copy()
                    condition_df_2_corr = condition_df_2.copy()

                # calculate mean score for texture across all participants
                condition_df_1_corr = condition_df_1_corr.sort_values(by=['Texture']).groupby('Texture').mean(numeric_only=True)

                # calculate mean score for texture across all participants
                condition_df_2_corr = condition_df_2_corr.sort_values(by=['Texture']).groupby('Texture').mean(numeric_only=True)

                # calculate spearmans rank correlation
                corr = stats.spearmanr(condition_df_1_corr['Mean ratio'], condition_df_2_corr['Mean ratio'])

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
    fig = plt.figure(constrained_layout=True, dpi=300, figsize=(6.85, 6))
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams["font.family"] = "Arial"
    gs = GridSpec(3, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[1, :3])
    ax3 = fig.add_subplot(gs[2, :3])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 3])
    ax6 = fig.add_subplot(gs[2, 3])

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
    df_roughness.groupby('Name').mean(numeric_only=True).sort_values(by='Mean ratio', ascending=False)['Mean ratio'].to_csv(
        '../processed_data/roughness_rank.csv')
    sorted = pd.read_csv('../processed_data/roughness_rank.csv')['Name']

    # plot mean rating for each participant
    sns.swarmplot(data=df_roughness, x="Name", y="Mean ratio", hue="Condition",
                  hue_order=['walking', 'sitting', 'hand'], dodge=True, alpha=.5, ax=ax1, order=sorted, size=1.5)
    # plot mean rating from all participants
    sns.pointplot(
        data=df_roughness, x="Name", y="Mean ratio", hue="Condition",
        hue_order=['walking', 'sitting', 'hand'], marker="-", ax=ax1, join=False,
        dodge=.5, markers="_", markersize=20, errorbar=('ci', 0),
        errwidth=0, order=sorted, legend=False)
    ax1.legend([], [], frameon=False)
    ax1.set_title('Smooth - Rough', fontsize=9)
    ax1.set_xticklabels(sorted, rotation=80)
    ax1.set_ylabel('Mean ratio', fontsize=8)
    ax1.set_xlabel('')
    ax1.set_ylim(0, 4.5)
    ax1.set_xticklabels([])
    sns.despine(ax=ax1)

    # identify significant Friedman's test and indicate on plot
    for i in range(len(sorted)):
        p = friedman[(friedman['Metric'] == 'roughness') & (friedman['Texture name'] == sorted[i])]
        p = np.array(p['p'])
        if p < .05:
            ax1.scatter(i, 4.25, marker='*', c='black', s=3)

    # kde plot to show spread of scores
    sns.kdeplot(data=df_roughness, y='Mean ratio', hue='Condition', hue_order=['walking', 'sitting', 'hand'], \
                ax=ax4, legend=False)
    ax4.set_title('Smooth - Rough', fontsize=8)
    ax4.set_ylim(0, 4.5)
    ax4.set_ylabel('')
    ax4.set_xlabel('')
    ax4.set_xticklabels([])
    sns.despine(ax=ax4)

    # ------------------------------------------------ Hardness ------------------------------------------------ #

    # extract hardness data
    df_hardness = df[df['Metric'] == 'hardness']

    # plot mean rating for each participant
    sns.swarmplot(data=df_hardness, x="Name", y="Mean ratio", hue="Condition",
                  hue_order=['walking', 'sitting', 'hand'], dodge=True, alpha=.5, ax=ax2, order=sorted, size=1.5)
    # plot mean rating from all participants
    sns.pointplot(
        data=df_hardness, x="Name", y="Mean ratio", hue="Condition",
        hue_order=['walking', 'sitting', 'hand'], marker="-", ax=ax2, join=False,
        dodge=.5, markers="_", markersize=20, errorbar=('ci', 0),
        errwidth=0, order=sorted, legend=False)
    ax2.legend([], [], frameon=False)
    ax2.set_title('Soft - Hard', fontsize=9)
    ax2.set_xticklabels(sorted, rotation=80)
    ax2.set_ylabel('Mean ratio', fontsize=8)
    ax2.set_xlabel('')
    ax2.set_xticklabels([])
    ax2.set_ylim(0, 4.5)
    sns.despine(ax=ax2)

    # identify significant Friedman's test and indicate on plot
    for i in range(len(sorted)):
        p = friedman[(friedman['Metric'] == 'hardness') & (friedman['Texture name'] == sorted[i])]
        p = np.array(p['p'])
        if p < .05:
            ax2.scatter(i, 4.25, marker='*', c='black', s=3)

    # kde plot to show spread of scores
    ax5.set_title('Soft - Hard', fontsize=8)
    sns.kdeplot(data=df_hardness, y='Mean ratio', hue='Condition', hue_order=['walking', 'sitting', 'hand'], \
                ax=ax5, legend=False)
    ax5.set_ylim(0, 4.5)
    ax5.set_ylabel('')
    ax5.set_xlabel('')
    sns.despine(ax=ax5)
    ax5.set_xticklabels([])

    # ------------------------------------------------ Slipperiness ------------------------------------------------ #

    # extract data for slipperiness
    df_slipperiness = df[df['Metric'] == 'slipperiness']

    # plot mean rating for each participant
    sns.swarmplot(data=df_slipperiness, x="Name", y="Mean ratio", hue="Condition",
        hue_order=['walking','sitting','hand'], dodge=True, alpha=.5, ax=ax3, order=sorted, size=1.5)
    # plot mean rating from all participants
    sns.pointplot(
        data=df_slipperiness, x="Name", y="Mean ratio", hue="Condition",
        hue_order=['walking','sitting','hand'], marker="-", ax=ax3, join=False,
        dodge=.55, markers="_", markersize=40, errorbar=('ci', 0), errwidth=0, order=sorted, legend=False)
    ax3.legend([], [], frameon=False)
    ax3.set_title('Slippery - Sticky', fontsize=9)
    ax3.set_xticklabels(sorted, rotation=80)
    ax3.set_ylabel('Mean ratio', fontsize=8)
    ax3.set_xlabel('Texture', fontsize=8)
    ax3.set_ylim(0, 4.5)
    sns.despine(ax=ax3)

    # identify significant Friedman's test and indicate on plot
    for i in range(len(sorted)):
        p = friedman[(friedman['Metric'] == 'slipperiness') & (friedman['Texture name'] == sorted[i])]
        p = np.array(p['p'])
        if p < .05:
            ax3.scatter(i, 4.25, marker='*', c='black', s=3)

    # kde plot to show spread of scores
    ax6.set_title('Slippery - Sticky')
    sns.kdeplot(data=df_slipperiness, y='Mean ratio', hue='Condition', hue_order=['walking', 'sitting', 'hand'], \
                ax=ax6, legend=False)
    ax6.set_ylim(0, 4.5)
    ax6.set_ylabel('')
    sns.despine(ax=ax6)

    # space subplots
    plt.tight_layout()
    # save figure
    plt.savefig('../individual_figures/mean_ratio_per_texture.png')
    plt.savefig('../individual_figures/mean_ratio_per_texture.svg', dpi=600)


def mean_ratio_spread(df):

    av = df.groupby(['Condition','Metric','Texture']).mean(numeric_only=True).to_csv('../processed_data/averaged_over_textures.csv')

    df_av = pd.read_csv('../processed_data/averaged_over_textures.csv')

    for metric in metrics:

        for condition in conditions:

            df_cond = df_av[(df_av['Metric'] == metric) & (df_av['Condition'] == condition)]

            print(metric, ' - ', condition, '- min: ', df_cond['Mean ratio'].min())
            print(metric, ' - ', condition, '- max: ', df_cond['Mean ratio'].max())

            print(metric, ' - ', condition, '- ratio: ', (df_cond['Mean ratio'].max() / df_cond['Mean ratio'].min()))


def mean_rank_textures(df):
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
    fig = plt.figure(constrained_layout=True, dpi=600, figsize=(6.85, 6))
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams["font.family"] = "Arial"
    gs = GridSpec(3, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[1, :3])
    ax3 = fig.add_subplot(gs[2, :3])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 3])
    ax6 = fig.add_subplot(gs[2, 3])

    # load in Friedman analysis
    friedman = pd.read_csv('../stats_output/friedman_rank.csv')

    # remove participant 12 hand condition
    df = df.loc[~((df['Condition'] == 'hand') & (df['Participant'] == 'PPT_012'))]

    # set color palette
    sns.set_palette(condition_colors)

    # ------------------------------------------------ Roughness ------------------------------------------------ #

    # extract roughness data
    df_roughness = df[df['Metric'] == 'roughness']

    # sort data from most rough to least rough and save, then reload
    df_roughness.groupby('Name').mean(numeric_only=True).sort_values(by='Mean ratio', ascending=False)['Mean ratio'].to_csv(
        '../processed_data/roughness_rank.csv')
    sorted = pd.read_csv('../processed_data/roughness_rank.csv')['Name']

    # plot mean rating for each participant
    sns.swarmplot(data=df_roughness, x="Name", y="Rank", hue="Condition",
                  hue_order=['walking', 'sitting', 'hand'], dodge=True, alpha=.5, ax=ax1, order=sorted, size=1.25)
    # plot mean rating from all participants
    sns.pointplot(
        data=df_roughness, x="Name", y="Rank", hue="Condition",
        hue_order=['walking', 'sitting', 'hand'], marker="-", ax=ax1, join=False,
        dodge=.57, markers="_", markersize=40, errorbar=('ci', 0), errwidth=0, order=sorted, legend=False, scale=.7)
    ax1.legend([], [], frameon=False)
    ax1.set_title('Smooth - Rough', fontsize=9)
    ax1.set_xticklabels(sorted, rotation=80)
    ax1.set_ylabel('Rank', fontsize=8)
    #ax1.set_xlabel('Texture')
    ax1.set_xticklabels(['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''])
    ax1.set_xlabel('')
    ax1.set_xlim(-.6, 15.6)
    ax1.set_ylim(0.8, 16)
    #ax1.set_yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    ax1.set_yticks([2,4,6,8,10,12,14,16])
    sns.despine(ax=ax1)


    # identify significant Friedman's test and indicate on plot
    for i in range(len(sorted)):
        p = friedman[(friedman['Metric'] == 'roughness') & (friedman['Texture name'] == sorted[i])]
        p = np.array(p['p'])
        #if p < .05:
            #ax1.scatter(i, 16.25, marker='*', c='black', s=3)

    # kde plot to show spread of scores
    sns.kdeplot(data=df_roughness, x='Mean ratio', hue='Condition', hue_order=['walking', 'sitting', 'hand'], \
                ax=ax4, legend=False)
    ax4.set_title('Smooth - Rough', fontsize=9)
    ax4.set_xlim(0, 4.5)
    ax4.set_xticks([0, 2, 4])
    sns.despine(ax=ax4)
    ax4.set_xticklabels('')
    ax4.scatter(2, 0.26, marker='*', c='black', s=3)
    ax4.set_ylim(0,0.27)
    ax4.set_xlabel('')
    ax4.set_ylabel('Density', fontsize=8)

    # ------------------------------------------------ Hardness ------------------------------------------------ #

    # extract hardness data
    df_hardness = df[df['Metric'] == 'hardness']

    # plot mean rating for each participant
    sns.swarmplot(data=df_hardness, x="Name", y="Rank", hue="Condition",
                  hue_order=['walking', 'sitting', 'hand'], dodge=True, alpha=.5, ax=ax2, order=sorted, size=1.25)
    # plot mean rating from all participants
    sns.pointplot(
        data=df_hardness, x="Name", y="Rank", hue="Condition",
        hue_order=['walking', 'sitting', 'hand'], marker="-", ax=ax2, join=False,
        dodge=.57, markers="_", markersize=40, errorbar=('ci', 0), errwidth=0, order=sorted, legend=False, scale=.7)
    ax2.legend([], [], frameon=False)
    ax2.set_title('Soft - Hard', fontsize=9)
    ax2.set_xticklabels(sorted, rotation=80)
    ax2.set_xticklabels(['','','','','','','','','','','','','','','',''])
    ax2.set_ylabel('Rank', fontsize=8)
    #ax2.set_xlabel('Texture')
    ax2.set_xlabel('')
    ax2.set_ylim(0.8, 16)
    ax2.set_xlim(-.6, 15.6)
    #ax2.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    ax2.set_yticks([2, 4, 6, 8, 10, 12, 14, 16])
    sns.despine(ax=ax2)

    # identify significant Friedman's test and indicate on plot
    for i in range(len(sorted)):
        p = friedman[(friedman['Metric'] == 'hardness') & (friedman['Texture name'] == sorted[i])]
        p = np.array(p['p'])
        #if p < .05:
            #ax2.scatter(i, 16.1, marker='*', c='black',s=3)

    # kde plot to show spread of scores
    ax5.set_title('Soft - Hard', fontsize=9)
    sns.kdeplot(data=df_hardness, x='Mean ratio', hue='Condition', hue_order=['walking', 'sitting', 'hand'], \
                ax=ax5, legend=False)
    ax5.set_xlim(0, 4.5)
    sns.despine(ax=ax5)
    ax5.set_xticks([0,2,4])
    ax5.set_xticklabels('')
    ax5.scatter(2, 0.26, marker='*', c='black', s=3)
    ax5.set_ylim(0, 0.27)
    ax5.set_xlabel('')
    ax5.set_ylabel('Density', fontsize=8)

    # ------------------------------------------------ Slipperiness ------------------------------------------------ #

    # extract data for slipperiness
    df_slipperiness = df[df['Metric'] == 'slipperiness']

    # plot mean rating for each participant
    sns.swarmplot(data=df_slipperiness, x="Name", y="Rank", hue="Condition",
        hue_order=['walking','sitting','hand'], dodge=True, alpha=.5, ax=ax3, order=sorted, size=1.25)
    # plot mean rating from all participants
    sns.pointplot(
        data=df_slipperiness, x="Name", y="Rank", hue="Condition",
        hue_order=['walking','sitting','hand'], marker="-", ax=ax3, join=False,
        dodge=.57, markers="_", markersize=40, errorbar=('ci', 0), errwidth=0, order=sorted, legend=False, scale=.7)
    ax3.legend([], [], frameon=False)
    ax3.set_title('Slippery - Sticky', fontsize=9)
    ax3.set_xticklabels(sorted, rotation=80)
    ax3.set_ylabel('Rank', fontsize=8)
    ax3.set_xlabel('Texture', fontsize=8)
    ax3.set_ylim(0.8, 16)
    print(ax3.get_xticks())
    ax3.set_xlim(-.6, 15.6)
    #ax3.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    ax3.set_yticks([2, 4, 6, 8, 10, 12, 14, 16])
    #ax3.set_xticks([])
    sns.despine(ax=ax3)

    # identify significant Friedman's test and indicate on plot
    for i in range(len(sorted)):
        p = friedman[(friedman['Metric'] == 'slipperiness') & (friedman['Texture name'] == sorted[i])]
        p = np.array(p['p'])
        #if p < .05:
            #ax3.scatter(i, 16.1, marker='*', c='black', s=3.)

    # kde plot to show spread of scores
    ax6.set_title('Slippery - Sticky', fontsize=9)
    sns.kdeplot(data=df_slipperiness, x='Mean ratio', hue='Condition', hue_order=['walking', 'sitting', 'hand'], \
                ax=ax6, legend=False)
    ax6.set_xlim(0, 4.5)
    sns.despine(ax=ax6)
    ax6.set_xticks([0,2,4])
    ax6.set_ylim(0, 0.27)
    ax6.set_xlabel('Rating', fontsize=8)
    ax6.set_ylabel('Density', fontsize=8)

    # space subplots
    #plt.tight_layout()
    plt.subplots_adjust(wspace=.25)
    # save figure
    plt.savefig('../individual_figures/mean_rank_per_texture.png')
    plt.savefig('../individual_figures/mean_rank_per_texture.svg', dpi=600, bbox_inches='tight')



def correlate_metrics_within_conditions(df):
    """ Run a correlation for each metric within each condition

    :param df:
    :return:
    """

    # only use the mean ratio per participant
    df = df[df['Trial'] == 2.0]

    # define plot parameters
    fig = plt.figure(constrained_layout=True, dpi=300, figsize=(8, 2))
    plt.rcParams.update({'font.size': 5})
    gs = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    axes = [ax1, ax2, ax3]

    i = 0
    # loop through conditions
    for condition in dimensions:

        # extract data for relevant condition
        condition_df = df[df['Condition'] == condition]

        # remove participant 12 hand condition
        if condition == 'hand':
            condition_df = condition_df[condition_df['Participant'] != 'PPT_012']

        # remove condition column
        condition_df.drop(columns=['Condition'])

        # create pivot table to rearrange data
        pivoted = pd.pivot_table(condition_df, values='Mean ratio', index=['Texture'],
                                 columns=['Metric'])

        # plot correlation matric
        sns.heatmap(pivoted.corr(method='spearman'), vmin=0., vmax=1., ax=axes[i], square=True, annot=True, fmt=".3f")
        axes[i].set_title(condition)

        i += 1

    # save plot
    plt.savefig('../individual_figures/single_value_per_texture_correlating_metrics_within_conditions.png')


    # ------------------------- Remove stability from correlation matrix ------------------------- #
    # define plot parameters
    fig = plt.figure(constrained_layout=True, dpi=300, figsize=(8, 2))
    plt.rcParams.update({'font.size': 5})
    gs = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    axes = [ax1, ax2, ax3]

    # remove stability
    no_stab = df[df['Metric'] != 'stability']

    i = 0
    # loop through conditions
    for condition in dimensions:

        # extract data for relevant condition
        condition_df = no_stab[no_stab['Condition'] == condition]

        # remove participant 12 hand condition
        if condition == 'hand':
            condition_df = condition_df[condition_df['Participant'] != 'PPT_012']

        # drop condition column
        condition_df.drop(columns=['Condition'])

        # create pivot table to reararnge data
        pivoted = pd.pivot_table(condition_df, values='Mean ratio', index=['Texture'],
                                 columns=['Metric'])

        # plot correlation matrix
        sns.heatmap(pivoted.corr(method='spearman'), vmin=0., vmax=1., ax=axes[i], square=True, annot=True, fmt=".3f")
        axes[i].set_title(condition)

        i += 1

    # save plot
    plt.savefig('../individual_figures/single_value_no_stability_correlating_metrics_within_conditions.png')

    # ---------------------------- run correlation using scipy ---------------------------- #
    # create dataframe to store result
    correlation_df = pd.DataFrame({'Condition': [], 'covariate1': [], 'covariate2': [], 'r':[], 'p': []})

    # loop through conditions
    for condition in dimensions:

        # extract data for relevant condition
        condition_df = df[df['Condition'] == condition]

        # remove participant 12 hand condition
        if condition == 'hand':
            condition_df = condition_df[condition_df['Participant'] != 'PPT_012']

        # loop through metrics
        for metric1 in dimensions[condition]:

            # extract data for relevant metric and sort by texture number
            metric_df_1 = condition_df[condition_df['Metric'] == metric1]
            metric_df_1 = metric_df_1.sort_values(by=['Texture'])

            # loop through metrics
            for metric2 in dimensions[condition]:

                # extract data for relevant metric and sort by texture number
                metric_df_2 = condition_df[condition_df['Metric'] == metric2]
                metric_df_2 = metric_df_2.sort_values(by=['Texture'])

                # run spearmans rank correlation
                corr = stats.spearmanr(metric_df_1['Mean ratio'], metric_df_2['Mean ratio'])

                # store data in dataframe
                df_single = pd.DataFrame({'Condition': [condition], 'covariate1': [metric1], 'covariate2': [metric2], \
                                          'r': [corr[0]], 'p': [corr[1]]})

                # join dataframes
                correlation_df = pd.concat([correlation_df, df_single])

    # save as .csv file
    correlation_df.to_csv('../stats_output/single_value_per_texture_correlating_metrics_within_conditions.csv')


def inter_participant_correlate_metrics_within_conditions(df):
    """ Run a correlation for each metric within each condition

    :param df:
    :return:
    """

    # only use the mean ratio per participant
    df = df[df['Trial'] == 2.0]

    # create dataframe to store result
    correlation_df = pd.DataFrame({'Condition': [], 'covariate1': [], 'covariate2': [], 'participant': [], 'r':[], 'p': []})

    # loop through conditions
    for condition in dimensions:

        # extract data for relevant condition
        condition_df = df[df['Condition'] == condition]

        # loop through participant ids
        for participant in participant_ids:

            # extract data for relevant participant
            ppt_df = condition_df[condition_df['Participant'] == participant]

            # loop through metrics
            for metric1 in dimensions[condition]:

                # extract data for relevant metric and sort by texture number
                metric_df_1 = ppt_df[ppt_df['Metric'] == metric1]
                metric_df_1 = metric_df_1.sort_values(by=['Texture'])

                # loop through metrics
                for metric2 in dimensions[condition]:

                    # extract data for relevant metric and sort by texture number
                    metric_df_2 = ppt_df[ppt_df['Metric'] == metric2]
                    metric_df_2 = metric_df_2.sort_values(by=['Texture'])

                    # do not run correlations for the hand for participant 12 as did not complete condition
                    if condition == 'hand' and participant == 'PPT_012':
                        continue
                    else:
                        # run spearmans rank correlation
                        corr = stats.spearmanr(metric_df_1['Mean ratio'], metric_df_2['Mean ratio'])

                        # store data in dataframe
                        df_single = pd.DataFrame({'Condition': [condition], 'covariate1': [metric1], \
                                                  'covariate2': [metric2],  'participant': [participant],\
                                                  'r': [corr[0]], 'p': [corr[1]]})

                        # join dataframes
                        correlation_df = pd.concat([correlation_df, df_single])

    # save result as .csv file
    correlation_df.to_csv('../stats_output/ppt_single_value_per_texture_correlating_metrics_within_conditions.csv')


def calculate_mean_ppt_level_correlations_metrics_within_conditions():
    """ Calculate mean correlation for each metric between conditions for each participant

        :return:
        """

    # load participant level correlations
    df = pd.read_csv('../stats_output/ppt_single_value_per_texture_correlating_metrics_within_conditions.csv')

    # create dataframe to store results
    mean_corr_df = pd.DataFrame({'Condition': [], 'covariate1': [], 'covariate2': [], 'mean r': [], 'std r': []})

    # loop through conditions
    for condition in dimensions:

        # extract data for relevant condition
        condition_df = df[df['Condition'] == condition]

        # loop through metrics
        for metric1 in dimensions[condition]:

            # loop through metrics
            for metric2 in dimensions[condition]:

                # extract data for relevant comparison
                comparison_df = condition_df[(condition_df['covariate1'] == metric1) & (condition_df['covariate2'] == metric2)]

                # calculate mean correlation
                # store in dataframe
                df_single = pd.DataFrame(
                    {'Condition': [condition], 'covariate1': [metric1], 'covariate2': [metric2], \
                     'mean r': comparison_df['r'].mean(), 'std r': comparison_df['r'].std()})

                # join dataframes
                mean_corr_df = pd.concat([mean_corr_df, df_single])

    # save result as .csv file
    mean_corr_df.to_csv('../stats_output/mean_ppt_correlation_metrics_within_conditions.csv')


def multiple_regression(df):
    """ Run a multiple regresison to calculate the contribution of each of the 3 textural dimensions on particpants'
    rating of stability.
    Plot metric relation to condition as scatterplots

    :param df:
    :param outcome_metric:
    :return:
    """

    # use only trial 2 so one value per texture per participant
    df = df[df['Trial'] == 2]

    # sort data
    df = df.sort_values(by=['Participant', 'Texture'])

    # extract a single value per participant for each metric
    stability = df[(df['Condition'] == 'walking') & (df['Metric'] == 'stability')].rename(
        columns={'Mean ratio': 'Stability'}).groupby(['Texture']).mean(numeric_only=True)['Stability'].values
    roughness = df[(df['Condition'] == 'walking') & (df['Metric'] == 'roughness')].rename(
        columns={'Mean ratio': 'Roughness'}).groupby(['Texture']).mean(numeric_only=True)['Roughness'].values
    hardness = df[(df['Condition'] == 'walking') & (df['Metric'] == 'hardness')].rename(
        columns={'Mean ratio': 'Hardness'}).groupby(['Texture']).mean(numeric_only=True)['Hardness'].values
    slipperiness = df[(df['Condition'] == 'walking') & (df['Metric'] == 'slipperiness')].rename(
        columns={'Mean ratio': 'Slipperiness'}).groupby(['Texture']).mean(numeric_only=True)['Slipperiness'].values

    # create simple dataframe with only necessary data
    df_new = pd.DataFrame(
        {'Roughness': roughness, 'Hardness': hardness, 'Slipperiness': slipperiness, 'Stability': stability})

    # define input measures
    X = df_new[['Roughness', 'Hardness', 'Slipperiness']]
    # define outcome measure
    Y = df_new[['Stability']]

    # run regression analysis
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    X = sm.add_constant(X)
    est = sm.OLS(Y, X).fit()
    res = est.summary()
    print(res)

    #def func(x, a, b, c):
    #    return a * np.exp(-b * x) + c  # a and d are redundant
    def func(m, x, c):
        return m * x + c  # a and d are redundant

    # define plot parameters
    plt.close('all')
    fig = plt.figure(constrained_layout=True, dpi=600, figsize=(3.14961, 3.14961))
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams["font.family"] = "Arial"
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])

    # plot scatterplot between roughness and stability
    sns.scatterplot(x=roughness, y=stability, color='limegreen', ax=ax1, s=15)
    sns.despine(ax=ax1)
    ax1.set_xlim(0, 3.5)
    ax1.set_ylim(0, 2)
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_yticks([0, 1, 2])
    ax1.set_ylabel('Mean ratio\nUnstable to Stable', fontsize=8)
    ax1.set_title('Roughness', fontsize=9)
    ax1.set_xlabel('Smooth to Rough', fontsize=8)
    popt, pcov = curve_fit(func, roughness, stability, maxfev=10000)
    x = np.arange(np.min(roughness), 3.5, 0.25)
    y = func(x, popt[0], popt[1])#, popt[2])#, popt[3])
    y[y < 0] = 0.
    ax1.plot(x, y, color='black', lw=1.)
    print('roughness-stability: ', stats.pearsonr(roughness, stability)[0] ** 2)
    print(stats.pearsonr(roughness, stability))

    print('----------------------------------------------------')

    # plot scatterplot between hardness and stability
    sns.scatterplot(x=hardness, y=stability, color='royalblue', ax=ax2, s=15)
    sns.despine(ax=ax2)
    ax2.set_xlim(0, 3.5)
    ax2.set_ylim(0, 2)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xlabel('Soft - hard', fontsize=8)
    ax2.set_title('Hardness', fontsize=9)
    ax2.set_xlabel('Soft to Hard', fontsize=8)
    popt, pcov = curve_fit(func, hardness, stability, maxfev=10000)
    x = np.arange(np.min(hardness), 3.5, 0.25)
    y = func(x, popt[0], popt[1])#, popt[2])#, popt[3])
    y[y < 0] = 0.
    ax2.plot(x, y, color='black', lw=1.)
    print('hardness-stability: ', stats.pearsonr(hardness, stability)[0] ** 2)
    print(stats.pearsonr(hardness, stability))

    print('----------------------------------------------------')

    # plot scatterplot between slipperiness and stability
    sns.scatterplot(x=slipperiness, y=stability, color='orangered', ax=ax3, s=15)
    sns.despine(ax=ax3)
    ax3.set_xlim(0, 3.5)
    ax3.set_ylim(0, 2)
    ax3.set_xticks([0, 1, 2, 3])
    ax3.set_yticks([0, 1, 2])
    #ax3.set_xlabel('Slippery - sticky', fontsize=8)
    #ax3.set_ylabel('Mean ratio\nUnstable to Stable', fontsize=8)
    ax3.set_ylabel('Mean ratio\nUnstable to Stable', fontsize=8)
    ax3.set_title('Stickiness', fontsize=9)
    ax3.set_xlabel('Slippery to Sticky\nMean ratio', fontsize=8)
    popt, pcov = curve_fit(func, slipperiness, stability, maxfev=10000)
    x = np.arange(np.min(slipperiness), 3.5, 0.25)
    y = func(x, popt[0], popt[1])#, popt[2])#, popt[3])
    y[y < 0] = 0.
    ax3.plot(x, y, color='black', lw=1.)
    print('stickiness-stability: ', stats.pearsonr(slipperiness, stability)[0]**2)
    print(stats.pearsonr(slipperiness, stability))

    print('----------------------------------------------------')

    # plot the final regression model
    predicted = .2272*roughness + .2891*hardness + -.4666*slipperiness + .9500
    sns.scatterplot(x=stability, y=predicted, color='black', ax=ax4, s=15)
    sns.despine(ax=ax4)
    ax4.set_xlim(0, 2)
    ax4.set_ylim(0, 2)
    ax4.set_xticks([0, 1, 2])
    ax4.set_yticks([0, 1, 2])
    ax4.set_ylabel('Actual rating', fontsize=8)
    ax4.set_xlabel('Unstable to Stable\nPredicted rating', fontsize=8)
    ax4.set_title('Final regression model', fontsize=9)
    popt, pcov = curve_fit(func, stability, predicted, maxfev=10000)
    x = np.arange(np.min(stability), 3.5, 0.25)
    y = func(x, popt[0], popt[1])#, popt[2])# , popt[3])
    y[y < 0] = 0.
    ax4.plot(x, y, color='black', lw=1.)

    print('predicted-actual: ', stats.pearsonr(predicted, stability)[0] ** 2)
    print(stats.pearsonr(predicted, stability))

    print('----------------------------------------------------')

    # save plot
    plt.savefig('../individual_figures/single_value_per_texture_relation_to_stability.png')
    plt.savefig('../individual_figures/single_value_per_texture_relation_to_stability.svg', dpi=600)


def between_condition_ppt_level_heatmap():
    """

    :return:
    """


    data = pd.read_csv('../stats_output/mean_ppt_correlation_metrics_between_conditions.csv')

    fig = plt.figure(constrained_layout=True, dpi=300, figsize=(8, 2))
    plt.rcParams.update({'font.size': 5})
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    axes = [ax1, ax2, ax3]

    k = 0

    conditions.reverse()

    for metric in metrics[:-1]:

        array = np.zeros((3,3))

        for i in range(len(conditions)):

            for j in range(len(conditions)):

                array[i,j] = data[(data['Metric'] == metric) & (data['covariate1'] == conditions[i]) & (data['covariate2'] == conditions[j])]['mean r']


        axes[k].set_title(metric)
        sns.heatmap(array.T, vmin=0., vmax=1., ax=axes[k], square=True, annot=True, fmt=".2f", xticklabels=conditions, yticklabels=conditions)

        k+=1

    plt.savefig('../figures/mean_ppt_corr_between_conditions.png')


def within_condition_ppt_level_heatmap():


    data = pd.read_csv('../stats_output/mean_ppt_correlation_metrics_within_conditions.csv')

    fig = plt.figure(constrained_layout=True, dpi=300, figsize=(8, 2))
    plt.rcParams.update({'font.size': 5})
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    axes = [ax1, ax2, ax3]

    k = 0

    metrics = ['hardness','roughness','slipperiness','stability']

    for condition in conditions:

        array = np.zeros((3,3))

        for i in range(len(metrics[:-1])):

            for j in range(len(metrics[:-1])):

                array[i,j] = data[(data['Condition'] == condition) & (data['covariate1'] == metrics[:-1][i]) & (data['covariate2'] == metrics[:-1][j])]['mean r']


        axes[k].set_title(condition)
        sns.heatmap(array, vmin=0., vmax=1., ax=axes[k], square=True, annot=True, fmt=".2f", xticklabels=metrics[:-1], yticklabels=metrics[:-1])

        k+=1

    plt.savefig('../figures/mean_ppt_corr_within_conditions.png')


def spread_of_scores_between_conditions(df):
    """ Run Levene's test for homogeneity of variance to compare distribution of scores

    :param df:
    :return:
    """

    df = df[df['Trial'] == 2.0] # only use one value per texture rating

    levenes_df = pd.DataFrame({'Metric': [], 'F': [], 'p': []})
    # loop through metrics
    for metric in dimensions['sitting']:

        # select data for relevant metric
        metric_df = df[df['Metric'] == metric]

        # extract data for each condition
        # remove PPT 012 from hand condition as did not take part in condition
        hand_df = metric_df[(metric_df['Condition'] == 'hand') & (metric_df['Participant'] != 'PPT_012')]
        sitting_df = metric_df[metric_df['Condition'] == 'sitting']
        walking_df = metric_df[metric_df['Condition'] == 'walking']

        # run levenes
        levenes = stats.levene(hand_df['Mean ratio'].values, sitting_df['Mean ratio'].values,
                               walking_df['Mean ratio'].values)

        # save levenes result as .csv file
        print(metric, ': ', levenes)

        levenes_df = pd.concat([levenes_df, pd.DataFrame({'Metric': [metric], 'F': [levenes[0]], 'p': [levenes[1]]})])

    levenes_df.to_csv('../stats_output/levenes_between_conditions.csv')


def tidy_rank_per_condition(df):
    """ Slope plot showing the rank of each texture across conditions

    :param df:
    :param texture:
    :param metric:
    :param condition1:
    :param condition2:
    :return:
    """

    df = df[df['Trial'] == 2] # only trial 2 so one value per participant and texture
    df = df[df['Participant'] != 'PPT_012'] # remove participant 12 as they did not complete hand conditions so cannot compare ranks

    plotting_df = pd.DataFrame({'Condition': [], 'Metric': [], 'Rank': [], 'Texture': []})

    # loop through metrics
    for metric in dimensions['sitting']:

        # extract only relevant metric
        metric_df = df[df['Metric'] == metric]

        # loop through conditions
        for condition in dimensions:

            # extract data for relevant condition
            condition_df = metric_df[metric_df['Condition'] == condition]

            # order the textures from low to high ratings
            ordered = condition_df.groupby('Texture').mean(numeric_only=True).sort_values(by='Mean ratio').reset_index()[
                'Texture'].values

            # add to dataframe
            ordered_df = pd.DataFrame(
                {'Condition': condition, 'Metric': metric, 'Rank': list(range(1, 17)), 'Texture': ordered})
            plotting_df = pd.concat([plotting_df, ordered_df])

    # reset indexes
    plotting_df = plotting_df.reset_index()

    # add column to store texture names
    plotting_df['Name'] = np.zeros(plotting_df.shape[0])

    # loop through textures
    for i in texture_names_simple:

        # find row indexes for each texture ID
        idxs = plotting_df[plotting_df['Texture'] == i].index

        # add texture name to texture ID
        plotting_df.loc[idxs, 'Name'] = texture_names_simple[i]


    fig = plt.figure(constrained_layout=True, dpi=600, figsize=(6.85, 4))
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams["font.family"] = "Arial"
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    sns.set_palette(list(texture_colors.values()))


    hardness = plotting_df[plotting_df['Metric'] == 'hardness']
    ax2.set_title('Hardness\n', fontsize=9)
    sns.pointplot(data=hardness, x='Condition', y='Rank', hue='Name', ax=ax2,
                  hue_order=list(texture_names_simple.values()), scale=0.5, capsize=.05,  # errorbar='sd',\
                  alpha=0.3)
    plt.setp(ax2.collections, alpha=.7, clip_on=False)  # for the markers
    plt.setp(ax2.lines, alpha=.3, clip_on=False)  # for the lines)
    ax2.set_ylim(0.8, 16)
    ax2.set_yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    sns.despine(ax=ax2)
    ax2.set_ylabel('Soft to Hard', fontsize=8)
    ax2.legend([], [], frameon=False)
    ax2.set_xticklabels(['Foot\nStepping', 'Foot\nExploration', 'Hand\nExploration'])

    roughness = plotting_df[plotting_df['Metric'] == 'roughness']
    ax1.set_title('Roughness\n', fontsize=9)
    sns.pointplot(data=roughness, x='Condition', y='Rank', hue='Name', ax=ax1,
                  hue_order=list(texture_names_simple.values()), scale=0.5, capsize=.05,  # errorbar='sd',\
                  alpha=0.3)
    plt.setp(ax1.collections, alpha=.7, clip_on=False)  # for the markers
    plt.setp(ax1.lines, alpha=.3, clip_on=False)  # for the lines)
    ax1.set_ylim(0.8, 16)
    ax1.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    sns.despine(ax=ax1)
    ax1.set_ylabel('Rank \n Smooth to Rough', fontsize=8)
    ax1.legend([], [], frameon=False)
    ax1.set_xticklabels(['Foot\nStepping', 'Foot\nExploration', 'Hand\nExploration'])

    slipperiness = plotting_df[plotting_df['Metric'] == 'slipperiness']
    ax3.set_title('Stickiness\n', fontsize=9)
    sns.pointplot(data=slipperiness, x='Condition', y='Rank', hue='Name', ax=ax3,
                  hue_order=list(texture_names_simple.values()), scale=0.5, capsize=.05,  # errorbar='sd',\
                  alpha=0.3)
    plt.setp(ax3.collections, alpha=.7, clip_on=False)  # for the markers
    plt.setp(ax3.lines, alpha=.3, clip_on=False)  # for the lines)
    ax3.set_ylim(0.8, 16)
    ax3.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    sns.despine(ax=ax3)
    ax3.set_ylabel('Slippery to Sticky', fontsize=8)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., title='Texture')
    ax3.set_xticklabels(['Foot\nStepping', 'Foot\nExploration', 'Hand\nExploration'])

    plt.subplots_adjust(wspace=.25)
    plt.savefig('../individual_figures/texture_rank_across_conditions.png', bbox_inches='tight')
    plt.savefig('../individual_figures/texture_rank_across_conditions.svg', bbox_inches='tight')


    fig = plt.figure(constrained_layout=True, dpi=600, figsize=(6.85, 4))
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams["font.family"] = "Arial"
    gs = GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    sns.set_palette(list(texture_colors.values()))


    hardness = plotting_df[plotting_df['Metric'] == 'hardness']
    ax2.set_title('Hardness\n', fontsize=9)
    sns.pointplot(data=hardness, x='Condition', y='Rank', hue='Name', ax=ax2,
                  hue_order=list(texture_names_simple.values()), scale=.83, capsize=.05,  # errorbar='sd',\
                  alpha=0.3, markers=['o','*','s','d','p','v','h','*','>','P','X','>','H','D','^','.'])
    plt.setp(ax2.collections, alpha=.7, clip_on=False)  # for the markers
    plt.setp(ax2.lines, alpha=.3, clip_on=False)  # for the lines)
    ax2.set_ylim(0.8, 16)
    ax2.set_yticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    sns.despine(ax=ax2)
    ax2.set_ylabel('Soft to Hard', fontsize=8)
    ax2.legend([], [], frameon=False)
    ax2.set_xticklabels(['Foot\nStepping', 'Foot\nExploration', 'Hand\nExploration'])

    roughness = plotting_df[plotting_df['Metric'] == 'roughness']
    ax1.set_title('Roughness\n', fontsize=9)
    sns.pointplot(data=roughness, x='Condition', y='Rank', hue='Name', ax=ax1,
                  hue_order=list(texture_names_simple.values()), scale=.83, capsize=.05,  # errorbar='sd',\
                  alpha=0.3, markers=['o','*','s','d','p','v','h','*','>','P','X','>','H','D','^','.'])
    plt.setp(ax1.collections, alpha=.7, clip_on=False)  # for the markers
    plt.setp(ax1.lines, alpha=.3, clip_on=False)  # for the lines)
    ax1.set_ylim(0.8, 16)
    ax1.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    sns.despine(ax=ax1)
    ax1.set_ylabel('Rank \n Smooth to Rough', fontsize=8)
    ax1.legend([], [], frameon=False)
    ax1.set_xticklabels(['Foot\nStepping', 'Foot\nExploration', 'Hand\nExploration'])

    slipperiness = plotting_df[plotting_df['Metric'] == 'slipperiness']
    ax3.set_title('Stickiness\n', fontsize=9)
    sns.pointplot(data=slipperiness, x='Condition', y='Rank', hue='Name', ax=ax3,
                  hue_order=list(texture_names_simple.values()), scale=.83, capsize=.05,  # errorbar='sd',\
                  alpha=0.3, markers=['o','*','s','d','p','v','h','*','>','P','X','>','H','D','^','.'])
    plt.setp(ax3.collections, alpha=.7, clip_on=False)  # for the markers
    plt.setp(ax3.lines, alpha=.3, clip_on=False)  # for the lines)
    ax3.set_ylim(0.8, 16)
    ax3.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    sns.despine(ax=ax3)
    ax3.set_ylabel('Slippery to Sticky', fontsize=8)
    ax3.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., title='Texture')
    ax3.set_xticklabels(['Foot\nStepping', 'Foot\nExploration', 'Hand\nExploration'])

    plt.subplots_adjust(wspace=.25)
    plt.tight_layout()
    plt.savefig('../individual_figures/texture_rank_across_conditions2.png', bbox_inches='tight')
    plt.savefig('../individual_figures/texture_rank_across_conditions2.svg', bbox_inches='tight')


def non_parametric_rm_ANOVA_rank(df):
    """ Run a non-parametric repeated measures ANOVA - Friedman's test
    To compare the mean ranks of each texture for a given metric across conditions

    :param df:
    :param metric:
    :return:
    """

    df = df[df['Trial'] == 2.]

    friedman_df = pd.DataFrame({'Metric': [], 'Texture number': [], 'Texture name': [], 'F': [], 'p': []})

    # remove participant 12 from analysis as did not complete the hand condition
    df = df[df['Participant'] != 'PPT_012']

    for metric in metrics[:-1]:
        metric_df = df[df['Metric'] == metric]

        metric_df.drop(columns=['Metric'])

        pivoted = pd.pivot_table(metric_df, values='Rank', index=['Texture', 'Participant'],
                                 columns=['Condition'])

        friedman_df_metric = pd.DataFrame({'Metric': [], 'Texture': [], 'F': [], 'p': []})

        for i in range(1,17):

            hand = pivoted.loc[i]['hand'].values
            sitting = pivoted.loc[i]['sitting'].values
            walking = pivoted.loc[i]['walking'].values

            friedman = stats.friedmanchisquare(hand, sitting, walking)

            texture_df = pd.DataFrame({'Metric': [metric], 'Texture number': [i], 'Texture name': [texture_names_simple[i]], 'F': [friedman[0]], 'p': [friedman[1]]})

            friedman_df_metric = pd.concat([friedman_df_metric, texture_df], ignore_index=True)


        friedman_df = pd.concat([friedman_df, friedman_df_metric], ignore_index=True)

    friedman_df.to_csv('../stats_output/friedman_rank.csv')


def non_parametric_rm_t_test_rank(df):
    """ Run a non-parametric repeated measures t-test - Wilcoxen signed rank test
    To compare the mean ranks of each texture for a given metric across conditions

    :param df:
    :param metric:
    :return:
    """

    df = df[df['Trial'] == 2.]

    wilcoxen_df = pd.DataFrame({'Metric': [], 'Texture number': [], 'Texture name': [], 'Comparison': [], 'W': [], 'p': []})

    # remove participant 12 from analysis as did not complete the hand condition
    df = df[df['Participant'] != 'PPT_012']

    # loop through metrics except stability
    for metric in metrics[:-1]:

        metric_df = df[df['Metric'] == metric]

        # print(pd.DataFrame(condition_df.groupby(['Texture', 'Metric']).mean()))

        metric_df.drop(columns=['Metric'])

        #pivoted = pd.pivot_table(metric_df, values='Ratio', index=['Texture', 'Participant'],
        #                         columns=['Condition'])
        pivoted = pd.pivot_table(metric_df, values='Rank', index=['Texture', 'Participant'],
                                 columns=['Condition'])


        wilcoxen_df_metric = pd.DataFrame({'Metric': [], 'Texture': [], 'Comparison': [], 'W': [], 'p': []})
        for i in range(1,17):

            hand = pivoted.loc[i]['hand'].values
            sitting = pivoted.loc[i]['sitting'].values
            walking = pivoted.loc[i]['walking'].values

            hand_sitting = stats.wilcoxon(hand, sitting)
            hand_walking = stats.wilcoxon(hand, walking)
            sitting_walking = stats.wilcoxon(sitting, walking)

            h_s_df = pd.DataFrame({'Metric': [metric], 'Texture number': [i], 'Texture name': [texture_names_simple[i]], \
                                   'Comparison': 'hand-sitting', 'W': [hand_sitting[0]], 'p': [hand_sitting[1]]})
            h_w_df = pd.DataFrame({'Metric': [metric], 'Texture number': [i], 'Texture name': [texture_names_simple[i]], \
                                   'Comparison': 'hand-walking', 'W': [hand_walking[0]], 'p': [hand_walking[1]]})
            s_w_df = pd.DataFrame({'Metric': [metric], 'Texture number': [i], 'Texture name': [texture_names_simple[i]], \
                                   'Comparison': 'sitting-walking', 'W': [sitting_walking[0]], 'p': [sitting_walking[1]]})

            wilcoxen_df_metric = pd.concat([wilcoxen_df_metric, h_s_df, h_w_df, s_w_df], ignore_index=True)

        print(wilcoxen_df)

        wilcoxen_df = pd.concat([wilcoxen_df, wilcoxen_df_metric], ignore_index=True)

    wilcoxen_df.to_csv('../stats_output/wilcoxen_rank.csv')


def texture_level_correlate_metrics_between_conditions_textures(df):
    """ Run a correlation for each metric between conditions

    :param df:
    :return:
    """

    df = df[df['Trial'] == 2.0] # only use mean ratio per participant

    # SINGLE VALUE PER TEXTRUE

    fig = plt.figure(constrained_layout=True, dpi=300, figsize=(15, 15))
    plt.rcParams.update({'font.size': 5})
    gs = GridSpec(3, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])
    ax6 = fig.add_subplot(gs[5])
    ax7 = fig.add_subplot(gs[6])
    ax8 = fig.add_subplot(gs[7])
    ax9 = fig.add_subplot(gs[8])

    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    i = 0

    comparisons = [['walking','sitting'], ['walking','hand'],['sitting','hand']]

    for metric in dimensions['sitting']:

        metric_df = df[df['Metric'] == metric]

        for comparison in comparisons:

            comparison_df = metric_df[(metric_df['Condition'] == comparison[0]) | (metric_df['Condition'] == comparison[1])]

            pivoted = pd.pivot_table(comparison_df, values='Rank', index=['Condition','Participant'],
                                 columns=['Texture'])

            sns.heatmap(pivoted.corr(method='spearman'), vmin=-1, vmax=1., ax=axes[i], square=True, annot=False, fmt=".2f", \
                        cmap='viridis', xticklabels=list(texture_names_simple.values()), yticklabels=list(texture_names_simple.values()))
            axes[i].set_title(metric + ': ' + comparison[0] + ' - ' + comparison[1])

            i+=1

    plt.savefig('../individual_figures/texture_level_correlations.png')



    # get p-values
    correlation_df = pd.DataFrame({'Metric': [], 'texture': [], 'covariate1': [], 'covariate2': [], 'r': [], 'p': []})
    for metric in dimensions['sitting']:

        metric_df = df[df['Metric'] == metric]

        for condition1 in dimensions:

            condition_df_1 = metric_df[metric_df['Condition'] == condition1]
            if condition1 == 'hand':# or condition2 == 'hand':
                condition_df_1 = condition_df_1[condition_df_1['Participant'] != 'PPT_012']
            condition_df_1 = condition_df_1.sort_values(by=['Texture','Participant'])


            for condition2 in dimensions:

                condition_df_2 = metric_df[metric_df['Condition'] == condition2]
                if condition2 == 'hand':# or condition2 == 'hand':
                    condition_df_2 = condition_df_2[condition_df_2['Participant'] != 'PPT_012']
                condition_df_2 = condition_df_2.sort_values(by=['Texture','Participant'])

                # remove participant 12 from any comparison involving the hand condition, which they did not complete
                if condition1 == 'hand' or condition2 == 'hand':
                    condition_df_1 = condition_df_1[condition_df_1['Participant'] != 'PPT_012']
                    condition_df_2 = condition_df_2[condition_df_2['Participant'] != 'PPT_012']

                for i in range(1,17):

                    condition_df_1_text = condition_df_1[condition_df_1['Texture'] == i]
                    condition_df_2_text = condition_df_2[condition_df_2['Texture'] == i]

                    corr = stats.spearmanr(condition_df_1_text['Mean ratio'], condition_df_2_text['Mean ratio'])


                    df_single = pd.DataFrame(
                        {'Metric': [metric], 'texture': [i], 'covariate1': [condition1], 'covariate2': [condition2], 'r': [corr[0]],
                         'p': [corr[1]]})

                    correlation_df = pd.concat([correlation_df, df_single])

    correlation_df.to_csv('../stats_output/texture_level_correlating_metrics_between_conditions.csv')

def correlation_comparison():


    new_df = pd.DataFrame({'Comparison': [], 'Data': [], 'r': []})

    df3 = pd.read_csv('../stats_output/all_inter_subject_correlations.csv')

    for dimension in ['roughness', 'hardness', 'slipperiness']:


        for condition in ['walking', 'sitting', 'hand']:
            inter_data = df3[(df3['Metric'] == dimension) & (df3['Condition'] == condition)]

            inter_df = pd.DataFrame(
                {'Comparison': [dimension + '-' + condition] * inter_data.shape[0],
                 'Data': ['inter'] * inter_data.shape[0], 'r': inter_data['r'], 'Dimension': [dimension] * inter_data.shape[0]})

            new_df = pd.concat([new_df, inter_df])


    df1 = pd.read_csv('../stats_output/ppt_single_value_per_texture_correlating_metrics_between_conditions2.csv')

    comparisons = [['walking', 'sitting'], ['walking', 'hand'], ['sitting', 'hand']]

    for dimension in ['roughness','hardness','slipperiness']:
        for comparison in comparisons:
            ppt_data = df1[(df1['Metric'] == dimension) & (df1['covariate1'] == comparison[0]) & (df1['covariate2'] == comparison[1])]

            ppt_df = pd.DataFrame({'Comparison': [dimension + '-' + comparison[0] + '-' + comparison[1]] * ppt_data.shape[0],
                                   'Data': ['participant'] * ppt_data.shape[0], 'r': ppt_data['r'], 'Dimension': [dimension] * ppt_data.shape[0]})

            new_df = pd.concat([new_df, ppt_df])


    df2 = pd.read_csv('../stats_output/texture_level_correlating_metrics_between_conditions.csv')

    for dimension in ['roughness','hardness','slipperiness']:
        for comparison in comparisons:
            txt_data = df2[(df2['Metric'] == dimension) & (df2['covariate1'] == comparison[0]) & (df2['covariate2'] == comparison[1])]

            txt_df = pd.DataFrame({'Comparison': [dimension + '-' + comparison[0] + '-' + comparison[1]] * txt_data.shape[0],
                                   'Data': ['textures'] * txt_data.shape[0], 'r': txt_data['r'], 'Dimension': [dimension] * txt_data.shape[0]})

            new_df = pd.concat([new_df, txt_df])

    plot_inter = new_df[new_df['Data'] == 'inter']
    plot_ppt = new_df[new_df['Data'] == 'participant']

    fig = plt.figure(constrained_layout=True, dpi=600, figsize=(6.85, 4))
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams["font.family"] = "Arial"
    gs = GridSpec(1, 2, figure=fig)

    palette = ['limegreen', 'limegreen', 'limegreen', 'royalblue', 'royalblue', 'royalblue', 'orangered', 'orangered',
               'orangered', 'limegreen', 'limegreen', 'limegreen', 'royalblue', 'royalblue', 'royalblue', 'orangered',
               'orangered', 'orangered']
    sns.set_palette(palette)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    sns.swarmplot(data=plot_inter, x='Comparison', y='r', ax=ax1, alpha=.5, size=1, hue='Comparison', \
                  order=['roughness-walking', 'roughness-sitting', 'roughness-hand', \
                         'hardness-walking', 'hardness-sitting', 'hardness-hand', \
                         'slipperiness-walking', 'slipperiness-sitting', 'slipperiness-hand'])
    sns.pointplot(data=plot_inter, x='Comparison', y='r', marker="-", join=False,
                  markers="_", markersize=70, errorbar=('ci', 0), errwidth=0, legend=False, scale=2, ax=ax1, hue='Comparison', \
                  order=['roughness-walking', 'roughness-sitting', 'roughness-hand', \
                         'hardness-walking', 'hardness-sitting', 'hardness-hand', \
                         'slipperiness-walking', 'slipperiness-sitting', 'slipperiness-hand'])
    ax1.set_xticklabels(labels=['Foot\nStepping', 'Foot\nExploration', 'Hand\nExploration', \
                                'Foot\nStepping', 'Foot\nExploration', 'Hand\nExploration', \
                                'Foot\nStepping', 'Foot\nExploration', 'Hand\nExploration'], rotation=80)
    ax1.set_xlabel('Roughness                 Hardness                    Stickiness\nCondition', fontsize=8)
    ax1.set_ylim(-.25, 1.0)
    ax1.legend([], [], frameon=False)
    sns.despine(ax=ax1)
    ax1.set_ylabel('Correlation coefficient', fontsize=8)

    sns.swarmplot(data=plot_ppt, x='Comparison', y='r', ax=ax2, alpha=.5, size=3, hue='Comparison', \
                  order=['roughness-walking-sitting', 'roughness-walking-hand', 'roughness-sitting-hand', \
                         'hardness-walking-sitting', 'hardness-walking-hand', 'hardness-sitting-hand', \
                         'slipperiness-walking-sitting', 'slipperiness-walking-hand', 'slipperiness-sitting-hand'])
    sns.pointplot(data=plot_ppt, x='Comparison', y='r', marker="-", join=False,
                  markers="_", markersize=70, errorbar=('ci', 0), errwidth=0, legend=False, scale=2, ax=ax2, hue='Comparison', \
                  order=['roughness-walking-sitting', 'roughness-walking-hand', 'roughness-sitting-hand', \
                         'hardness-walking-sitting', 'hardness-walking-hand', 'hardness-sitting-hand', \
                         'slipperiness-walking-sitting', 'slipperiness-walking-hand', 'slipperiness-sitting-hand'])
    ax2.set_xticklabels(labels=['Foot           Foot\nStepping - Exploration', 'Foot           Hand\nStepping - Exploration', 'Foot           Hand\nExploration - Exploration', \
                                'Foot           Foot\nStepping - Exploration', 'Foot           Hand\nStepping - Exploration', 'Foot           Hand\nExploration - Exploration', \
                                'Foot           Foot\nStepping - Exploration', 'Foot           Hand\nStepping - Exploration', 'Foot           Hand\nExploration - Exploration'], rotation=80)
    ax2.set_xlabel('Roughness                 Hardness                    Stickiness\nComparison', fontsize=8)
    sns.despine(ax=ax2)
    ax2.legend([], [], frameon=False)
    ax2.set_ylim(-.25, 1.0)
    ax2.set_yticks([])
    ax2.set_ylabel('')
    ax2.spines["left"].set_visible(False)

    plt.subplots_adjust(wspace=0)
    plt.savefig('../individual_figures/correlation_comparison.png', bbox_inches='tight', dpi=600)
    plt.savefig('../individual_figures/correlation_comparison.svg', bbox_inches='tight', dpi=600)

    fig = plt.figure(constrained_layout=True, dpi=600, figsize=(6.85, 4))
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams["font.family"] = "Arial"
    gs = GridSpec(1, 1, figure=fig)

    palette = ['limegreen', 'limegreen', 'limegreen', 'royalblue', 'royalblue', 'royalblue', 'orangered', 'orangered',
               'orangered', 'limegreen', 'limegreen', 'limegreen', 'royalblue', 'royalblue', 'royalblue', 'orangered',
               'orangered', 'orangered']
    sns.set_palette(palette)

    ax1 = fig.add_subplot(gs[0])

    sns.swarmplot(data=plot_ppt, x='Comparison', y='r', ax=ax1, alpha=.5, size=4, hue='Comparison', \
                  order=['roughness-walking-sitting', 'roughness-walking-hand', 'roughness-sitting-hand', \
                         'hardness-walking-sitting', 'hardness-walking-hand', 'hardness-sitting-hand', \
                         'slipperiness-walking-sitting', 'slipperiness-walking-hand', 'slipperiness-sitting-hand'])
    sns.pointplot(data=plot_ppt, x='Comparison', y='r', marker="-", join=False,
                  markers="_", markersize=70, errorbar=('ci', 0), errwidth=0, legend=False, scale=2, ax=ax1, hue='Comparison', \
                  order=['roughness-walking-sitting', 'roughness-walking-hand', 'roughness-sitting-hand', \
                         'hardness-walking-sitting', 'hardness-walking-hand', 'hardness-sitting-hand', \
                         'slipperiness-walking-sitting', 'slipperiness-walking-hand', 'slipperiness-sitting-hand'])

    sns.set_palette(['#ff218c','#21b1ff','#f2d602', '#ff218c','#21b1ff','#f2d602', '#ff218c','#21b1ff','#f2d602'])
    sns.pointplot(data=plot_inter, x='Comparison', y='r', marker="-", join=False,
                  markers="x", markersize=25, errorbar=('ci', 0), errwidth=0, legend=False, scale=1.5, ax=ax1, hue='Comparison', \
                  order=['roughness-walking', 'roughness-sitting', 'roughness-hand', \
                         'hardness-walking', 'hardness-sitting', 'hardness-hand', \
                         'slipperiness-walking', 'slipperiness-sitting', 'slipperiness-hand'])
    ax1.set_xticklabels(labels=['Foot           Foot\nStepping - Exploration', 'Foot           Hand\nStepping - Exploration', 'Foot           Hand\nExploration - Exploration', \
                                'Foot           Foot\nStepping - Exploration', 'Foot           Hand\nStepping - Exploration', 'Foot           Hand\nExploration - Exploration', \
                                'Foot           Foot\nStepping - Exploration', 'Foot           Hand\nStepping - Exploration', 'Foot           Hand\nExploration - Exploration'], rotation=80)
    ax1.set_xlabel('Roughness                 Hardness                    Stickiness\nCondition', fontsize=8)
    ax1.set_ylim(0, 1.0)
    ax1.legend([], [], frameon=False)
    sns.despine(ax=ax1)
    ax1.set_ylabel('Correlation coefficient', fontsize=8)


    #plt.subplots_adjust(wspace=0)
    plt.savefig('../individual_figures/correlation_comparison2.png', bbox_inches='tight', dpi=600)
    plt.savefig('../individual_figures/correlation_comparison2.svg', bbox_inches='tight', dpi=600)

    fig = plt.figure(constrained_layout=True, dpi=600, figsize=(4.5, 4))
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams["font.family"] = "Arial"
    gs = GridSpec(1, 3, figure=fig)

    palette = ['limegreen', 'limegreen', 'limegreen', 'royalblue', 'royalblue', 'royalblue', 'orangered', 'orangered',
               'orangered', 'limegreen', 'limegreen', 'limegreen', 'royalblue', 'royalblue', 'royalblue', 'orangered',
               'orangered', 'orangered']
    sns.set_palette(palette)

    plot_df = pd.concat([plot_inter, plot_ppt])
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])

    sns.swarmplot(data=plot_ppt, x='Comparison', y='r', ax=ax1, alpha=.5, size=4, hue='Comparison', \
                  order=['roughness-walking-sitting', 'roughness-walking-hand', 'roughness-sitting-hand', \
                         'hardness-walking-sitting', 'hardness-walking-hand', 'hardness-sitting-hand', \
                         'slipperiness-walking-sitting', 'slipperiness-walking-hand', 'slipperiness-sitting-hand'])
    sns.pointplot(data=plot_ppt, x='Comparison', y='r', marker="-", join=False,
                  markers="_", markersize=70, errorbar=('ci', 0), errwidth=0, legend=False, scale=2, ax=ax1, hue='Comparison', \
                  order=['roughness-walking-sitting', 'roughness-walking-hand', 'roughness-sitting-hand', \
                         'hardness-walking-sitting', 'hardness-walking-hand', 'hardness-sitting-hand', \
                         'slipperiness-walking-sitting', 'slipperiness-walking-hand', 'slipperiness-sitting-hand'])
    sns.despine(ax=ax1)
    ax1.set_xticklabels(labels=['Foot           Foot\nStepping - Exploration', 'Foot           Hand\nStepping - Exploration', 'Foot           Hand\nExploration - Exploration', \
                                'Foot           Foot\nStepping - Exploration', 'Foot           Hand\nStepping - Exploration', 'Foot           Hand\nExploration - Exploration', \
                                'Foot           Foot\nStepping - Exploration', 'Foot           Hand\nStepping - Exploration', 'Foot           Hand\nExploration - Exploration'], rotation=80)
    ax1.set_xlabel('Roughness                 Hardness                    Stickiness\nCondition', fontsize=8)
    ax1.set_ylim(0, 1.0)
    ax1.legend([], [], frameon=False)
    ax1.set_ylabel('Correlation coefficient', fontsize=8)

    palette2 = ['limegreen', 'royalblue', 'orangered']
    sns.set_palette(palette2)
    sns.pointplot(data=plot_ppt, x='Dimension', y='r', marker='-', join=False,
                  markers='x', markersize=25, errorbar=('ci', 0), errwidth=0, legend=False, scale=1, ax=ax2, hue='Dimension',\
                  order=['roughness', 'hardness', 'slipperiness', 'dummy'])
    sns.pointplot(data=plot_inter, x='Dimension', y='r', marker='-', join=False,
                  markers='o', markersize=25, errorbar=('ci', 0), errwidth=0, legend=False, scale=1, ax=ax2, hue='Dimension',\
                  order=['roughness', 'hardness', 'slipperiness','dummy'])
    ax2.set_ylim(0.5, 1.0)
    ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_xticklabels(labels=['Roughness','Hardness','Stickiness', ''], rotation=80)
    ax2.set_xlabel('Dimension', fontsize=8)
    ax2.legend([], [], frameon=False)
    sns.despine(ax=ax2)
    ax2.set_ylabel('Correlation coefficient', fontsize=8)

    plt.subplots_adjust(wspace=.5)
    plt.savefig('../individual_figures/correlation_comparison3.png', bbox_inches='tight', dpi=600)
    plt.savefig('../individual_figures/correlation_comparison3.svg', bbox_inches='tight', dpi=600)



def metric_legend():
    fig = plt.figure(constrained_layout=True, dpi=600, figsize=(6.85, 6.85))
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams["font.family"] = "Arial"
    gs = GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0])
    sns.scatterplot(x=[1], y=[1], c=['limegreen'], label='Roughness')
    sns.scatterplot(x=[1], y=[1], c=['royalblue'], label='Hardness')
    sns.scatterplot(x=[1], y=[1], c=['orangered'], label='Stickiness')
    plt.legend(title='Perceptual dimension')
    plt.savefig('../individual_figures/metric_legend.svg', dpi=600)

def data_legend():
    fig = plt.figure(constrained_layout=True, dpi=600, figsize=(6.85, 6.85))
    plt.rcParams.update({'font.size': 6.5})
    plt.rcParams["font.family"] = "Arial"
    gs = GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0])
    sns.scatterplot(x=[1], y=[1], ax=ax, marker='x', label='Between-participants')
    sns.scatterplot(x=[1], y=[1], ax=ax, marker='o', label='Between-conditions')
    plt.legend(title='Correlation')
    plt.savefig('../individual_figures/data_legend.svg', dpi=600)


def correlation_comparison_stats():

    inter_ppt = pd.read_csv('../stats_output/all_inter_subject_correlations.csv')

    roughness_ppt = inter_ppt[inter_ppt['Metric'] == 'roughness']
    roughness_ppt_mean_r = roughness_ppt['r'].mean()
    hardness_ppt = inter_ppt[inter_ppt['Metric'] == 'hardness']
    hardness_ppt_mean_r = hardness_ppt['r'].mean()
    slipperiness_ppt = inter_ppt[inter_ppt['Metric'] == 'slipperiness']
    slipperiness_ppt_mean_r = slipperiness_ppt['r'].mean()


    inter_condition = pd.read_csv('../stats_output/ppt_single_value_per_texture_correlating_metrics_between_conditions2.csv')
    inter_condition = inter_condition[inter_condition['covariate1'] != inter_condition['covariate2']]
    inter_condition = inter_condition[((inter_condition['covariate1'] == 'walking') & (inter_condition['covariate2'] == 'sitting')) | \
                                      ((inter_condition['covariate1'] == 'walking') & (inter_condition['covariate2'] == 'hand')) | \
                                      ((inter_condition['covariate1'] == 'sitting') & (inter_condition['covariate2'] == 'hand'))]

    roughness_cond = inter_condition[inter_condition['Metric'] == 'roughness']
    roughness_cond_mean_r = roughness_cond['r'].mean()
    hardness_cond = inter_condition[inter_condition['Metric'] == 'hardness']
    hardness_cond_mean_r = hardness_cond['r'].mean()
    slipperiness_cond = inter_condition[inter_condition['Metric'] == 'slipperiness']
    slipperiness_cond_mean_r = slipperiness_cond['r'].mean()

    print("------------------------------------------------------------------------------------------------")
    print('differences in correlations of roughness between-participants and between-conditions: ',
          independent_corr(roughness_ppt_mean_r, roughness_cond_mean_r, roughness_ppt.shape[0], roughness_cond.shape[0], method='fisher'))

    print("------------------------------------------------------------------------------------------------")
    print('differences in correlations of hardness between-participants and between-conditions: ',
          independent_corr(hardness_ppt_mean_r, hardness_cond_mean_r, hardness_ppt.shape[0], hardness_cond.shape[0], method='fisher'))

    print("------------------------------------------------------------------------------------------------")
    print('differences in correlations of slipperiness between-participants and between-conditions: ',
          independent_corr(slipperiness_ppt_mean_r, slipperiness_cond_mean_r, slipperiness_ppt.shape[0], slipperiness_cond.shape[0], method='fisher'))

    print("------------------------------------------------------------------------------------------------")


def scores_over_trials(df):

    for condition in dimensions:

        for metric in dimensions[condition]:

            if condition == 'walking' and metric == 'slipperiness':
                continue
            else:

                df_combination = df[(df['Condition'] == condition) & (df['Metric'] == metric)]

                plt.figure(figsize=(12,12))

                for i in range(1,16):
                    plt.suptitle(condition + ' - ' + metric)
                    plt.subplot(4,4,i)
                    plt.title('Texture: ' + texture_names_simple[i])
                    df_texture = df_combination[df_combination['Texture'] == i]
                    #df_texture['Ratio'] = np.log(df_texture['Ratio'])
                    ax = sns.pointplot(data=df_texture, x='Trial', y='Ratio', hue='Participant', alpha=.4, legend=False)
                    plt.setp(ax.collections, alpha=.3)  # for the markers
                    plt.setp(ax.lines, alpha=.3)
                    plt.legend([], [], frameon=False)
                    plt.ylim(0,7)
                    plt.axhline(y=1, c='r')
                plt.tight_layout()
                plt.savefig('../figures/' + condition + '_' + metric + '_ratings_across_trials.png')