import matplotlib
matplotlib.use("Qt5Agg")
#matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cytoflow as flow
import timeit

import MetadataHelper
import utility as util

import re
import os

import pandas as pd

from LiveDeadReport import LiveDeadReport
from SurfaceReport import SurfaceReport
from PipelineHelper import executeXDGating

if __name__ == '__main__':
    flow.set_default_scale("logicle")
    plt.style.use('astroml')
    DATE = '05092018'
    config_name = 'controls_only'
    settings_path = "config/exp_{0!s}_config.yml".format(config_name)
    fs = False


    print("Running exp {0!s}...".format(config_name))
    start = timeit.default_timer()
    print("Loading settings...")
    settings = MetadataHelper.loadSettings(settings_path)
    print("Settings loaded!")
    pickle_path = "pickles/{}_flow_stats.pickle".format(settings['EXP_NAME'])
    print("Loading existing flow gates...")
    all_stats = MetadataHelper.loadExistingFlowGates(pickle_path)
    all_stats = False
    if all_stats is False:
        print("No exisiting gating found, generating new gates...")
        fc_list = executeXDGating(settings, save=True, load=False)
        print("Gating completed!")
        
        all_stats = util.get_allstats_table(fc_list, settings)

    # Process expansion data
    s0 = util.import_s0_expansion()
    #exp_number = re.findall('\d+', settings['EXP_NAME'])[0]
    exp_number = 99
    s1_raw, s1_day_fc, s1_fc_prod = util.import_s1_expansion(exp_number)
    tables_directory = "reports/{}/tables".format(settings['EXP_NAME'])

    ### Write Tables ###
    if not os.path.exists(tables_directory):
        try:
            os.makedirs(tables_directory)
        except OSERROR as e:
            if e.errno != errno.EEXIST:
                raise
                
    if s1_raw is not None:
        s0_parsed = util.get_s0_results(s1_raw,s0)

        s0_s1_fc_CD34 = util.get_s0_fc('S0 Fold Expansion VPA CD34+', s0_parsed, s1_fc_prod)
        s0_s1_fc_TNC = util.get_s0_fc('S0 Fold Expansion VPA TNC', s0_parsed, s1_fc_prod)
        s0_s1_fc_CD34_CD90 = util.get_s0_fc('S0 Fold Expansion VPA CD34+ CD90+', s0_parsed, s1_fc_prod)
        cumul_prod_list = [s0_s1_fc_TNC, s0_s1_fc_CD34, s0_s1_fc_CD34_CD90]
        s1_prod_list = [s1_fc_prod]
        
        file_path = "{}/{}_TNC_FCs.tsv".format(tables_directory, settings['EXP_NAME'])
        table_dict = {'s0_s1_fc_TNC': s0_s1_fc_TNC,
                      's0_s1_fc_CD34':s0_s1_fc_CD34,
                      's0_s1_fc_CD34_CD90': s0_s1_fc_CD34_CD90,
                      's1_fc_prod': s1_fc_prod}

        util.write_tables(file_path,table_dict)

    else:
        s0_parsed = s0
        cumul_prod_list = [s0_parsed]

    if s1_raw is not None:
        header = ['PRE-EXPANSION DAY', 'CB Number', 'Treatment']
    else:
        header = ['PRE-EXPANSION DAY', 'CB Number', 'Treatment']
        empty_df = all_stats[['PRE-EXPANSION DAY', 'CB Number', 'Treatment']]
        cumul_prod_list = [empty_df]
        s0_parsed = util.get_s0_results(empty_df, s0)

    all_stats = all_stats.merge(s0_parsed, how='left', on=header)
    all_stats['Day'] = all_stats['Day'].astype(int)

    all_stats['CD41p_CD42p'] = all_stats['CD41p_CD42p'].astype(float)
    all_stats['CD41p'] = all_stats['CD41p'].astype(float)
    all_stats['viability'] = all_stats['viability'].astype(float)
    all_stats.to_csv('all_stats_' +settings['EXP_NAME'] + '.csv')
    # calculate for MK, viable MK produced per total nucleated cell
    if s1_raw is not None:
        combined_prod = util.get_production_perMK(cumul_prod_list,header, all_stats)
        combined_prod = util.get_production_perMK(s1_prod_list, header, combined_prod, s0_name = ['S1 TNC'])
        combined_prod['Viable Cumul 41p42p MK per S1 CD34'] =  combined_prod['Viable Cumul 41p42p MK per S1 TNC']/(combined_prod['S0 Percentage VPA CD34+']*0.01)
        combined_prod['Viable Cumul 41p42p MK per S1 CD34 CD90'] =  combined_prod['Viable Cumul 41p42p MK per S1 TNC']/(combined_prod['S0 Percentage VPA CD34+ CD90+']*0.01)

        header = ['CB Number','PRE-EXPANSION DAY', 'Treatment','Day']
        file_path = "{}/{}_MK_FCs.tsv".format(tables_directory,settings['EXP_NAME'])
        
        table_dict = {'s0_s1_fc_TNC': combined_prod[header + ['Viable Cumul 41p42p MK per S1 TNC']],
                      's0_s1_fc_CD34':combined_prod[header + ['Viable Cumul 41p42p MK per S1 CD34']],
                      's0_s1_fc_CD34_CD90': combined_prod[header + ['Viable Cumul 41p42p MK per S1 CD34 CD90']],
                      's1_fc_prod': combined_prod[header + ['Viable Cumul 41p42p MK per S0 TNC']]}

        util.write_tables(file_path,table_dict)

    else:
        combined_prod = all_stats


    agg_df = pd.DataFrame()
    header = ['PRE-EXPANSION DAY', 'CB Number', 'Treatment']
    treatment_g = combined_prod.groupby(header)
    for key, item in treatment_g:
        t_df = treatment_g.get_group(key)
        # 'E5 CB500 #500 StemSpan D7'
        treatment_row = t_df.pivot(columns = 'Day').mean(axis=0)
        treatment_row.name = key
        agg_df = pd.concat([agg_df, treatment_row],axis=1)

    col_names = agg_df.transpose().columns
    not_s0_col_names = [x for x in col_names if not x[0].startswith("S0 ")]
    s0_col_names = [x[0] for x in col_names if x[0].startswith("S0 ")]
    s0_col_names = list(set(s0_col_names))
    ## Get all S0 columns

    new_agg_df = agg_df.transpose()[not_s0_col_names]
    # average the lists
    for cname in s0_col_names:
        col_subset = [x for x in col_names if x[0] is cname]
        new_col = agg_df.transpose()[col_subset].mean(axis=1)
        new_col.name = (cname,0)
        new_agg_df = pd.concat([new_agg_df, new_col], axis=1)

    cname2 = ['CD34p', 'CD41p', 'CD41p_CD42p', 'CD41p_CD42n', 'S1 TNC']
    col_subset2 = [x for x in col_names if x[0] in cname2]
    col_subset2 = col_subset2 + [x for x in col_names if 'Viable' in x[0]]

    stat_names = list(set([x[0] for x in col_subset2]))
    for sname in stat_names:
        col_subset3 = [x for x in col_names if x[0] is sname]
        new_col2 = agg_df.transpose()[col_subset3].max(axis=1)
        new_col2.name = ('Max '+ sname,0)
        new_agg_df = pd.concat([new_agg_df, new_col2], axis=1)
    
    new_agg_df.reset_index(inplace=True)
    new_agg_df[header]= new_agg_df['index'].apply(pd.Series)
    new_agg_df.dropna(axis=1, how='all', inplace=True)
    new_agg_df.to_csv('{}/{}_stats.tsv'.format(tables_directory,settings['EXP_NAME']), sep='\t', index=False)

    stop = timeit.default_timer()
    print(stop-start, " to process exp ", str(config_name))
