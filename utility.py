import os
import pdb
import re
from itertools import groupby, product

import fcsparser
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde

import cytoflow as flow
import cytoflow.utility as cutil

def get_allstats_table(fc_list, settings):
    conditions_n = len(fc_list)
    all_stats = pd.DataFrame()
    for fc in fc_list:
        for mexp in fc.metaexps:
            stats = {}
            stats['CB Number'],stats['Day'], stats['S0_period'], stats['Treatment'] = mexp.com
            stats['Miller Exp Number'] = "Exp_{0!s}".format(mexp.EXP_NAME)
            # save each item in a table
            stats = dict(stats.items() + mexp.stats.items())
            all_stats = all_stats.append(stats, ignore_index=True)

    all_stats.to_pickle("pickles/{0!s}_flow_stats.pickle".format(settings['EXP_NAME']))
    return(all_stats)

def write_tables(file_path, table_dict):

    with open(file_path, 'w') as infile:
        infile.write("P1 Cumulatiave Fold Change \n")
    with open(file_path, 'a') as infile:
        table_dict['s1_fc_prod'].to_csv(infile, sep="\t")
        infile.write("\nTNC Cumulatiave Fold Change (TNC) \n")
        table_dict['s0_s1_fc_TNC'].to_csv(infile, sep="\t")
        infile.write("\nTNC Cumulatiave Fold Change (CD34)\n")
        table_dict['s0_s1_fc_CD34'].to_csv(infile, sep="\t")
        infile.write("\nTNC Cumulatiave Fold Change (CD34 CD90)\n")
        table_dict['s0_s1_fc_CD34_CD90'].to_csv(infile, sep="\t")

def consolidate_settings(user_settings, exp_settings):
    """
    Pulls user settings and experiment settings from respective files. Experiment settings override user settings.

    Parameters
    ----------
    user_settings : str
    exp_settings : str

    Returns
    _______
    settings : dict
    """
    with open(user_settings, 'r') as f:
        user_dict = yaml.load(f)
    with open(exp_settings, 'r') as f:
        exp_dict = yaml.load(f)

    # overlapping arguments/keys are overridden with new value from the experiment
    combined = user_dict.copy()
    combined.update(exp_dict)
    return(combined)

def point_slope(x1,y1, x2,y2):
    slope = (y2-y1)/float(x2-x1)
    return slope

def get_settings(exp_num, user="default"):
    exp_settings = "config/exp_{0!s}_config.yml".format(exp_num)
    user_settings = "config/user_{}.yml".format(user)
    settings = consolidate_settings(user_settings, exp_settings)
    return(settings)

def elbow_criteria(x,y):
    x = np.array(x)
    y = np.array(y)
    # Slope between elbow endpoints

    if y[1] > y[0]:
        x = np.delete(x, 0)
        y = np.delete(y, 0)
    m1 = point_slope(x[0], y[0], x[-1], y[-1])
    # Intercept
    b1 = y[0] - m1*x[0]

    # Slope for perpendicular lines
    m2 = -1/m1

    # Calculate intercepts for perpendicular lines that go through data point
    b_array = y-m2*x
    x_perp = (b_array-b1)/(m1-m2)
    y_perp = m1*x_perp+b1

    # Calculate where the maximum distance to a line connecting endpoints is
    distances = np.sqrt((x_perp-x)**2+(y_perp-y)**2)
    index_max = np.where(distances==np.max(distances))[0][0]
    elbow_x = x[index_max]
    elbow_y = y[index_max]
    return elbow_x, elbow_y


def generate_isotypes(combos, iter_param, all_gated_data, EXP_NAME):
    isotype_table = pd.DataFrame()
    for com_number,com in enumerate(combos):

        query_str=get_query(com, iter_param)

        gated_subset = all_gated_data.query(query_str)
        iso_exp_subset = gated_subset.query('Isotype == True and Live == True')
        exp_subset = gated_subset.query('Isotype == False and Live == True')

        condition_entry = dict(zip(['Exp_'+x for x in iter_param.keys()], com))
        condition_entry.update(dict(zip(['Iso_'+x for x in iter_param.keys()], com)))

        condition_entry.update({"N_Exp": len(exp_subset.data), "N_Iso": len(iso_exp_subset.data)})
        isotype_table = isotype_table.append(pd.Series(condition_entry), ignore_index = True)

# Save the isotype table as a tsv file.
    isotype_table.to_csv("data/"+EXP_NAME+'_isotype_table.txt',sep='\t', index=False)
    return(isotype_table)


def get_channel_data(gated_comp, x_channel, y_channel):
    xscale = cutil.scale_factory('logicle', gated_comp, x_channel)
    yscale = cutil.scale_factory('logicle', gated_comp, y_channel)
    x = xscale(gated_comp[x_channel].values)
    y = yscale(gated_comp[y_channel].values)
    #x = gated_comp[x_channel].values
    #y = gated_comp[y_channel].values
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    #idx = z.argsort()
    #x,y,z = x[idx], y[idx], z[idx]
    return(x,y,z)

def generate_indices(exp, iter_param, x='Day'):
    """Generate indices based on an experiment and how it iterates
    Args:
        exp (exp object): experiment with treatment column and day column

    Returns:
        list of tuples describing the order in which the treatment, day combos will be placed on a grid

    """
    conditions = 1
    for key in iter_param.keys():
        if x not in key:
            conditions = len(exp.data[key].unique()) * conditions

    treatment_n = conditions
    day_n = len(exp.data['Day'].unique())

    ind_list =[]
    for x in range(day_n):
        for y in range(treatment_n):
            ind_list.append((x,y))

    return(ind_list)

def make_surface_group(metaexp_list, day = 7):
    """
    sets a reference surface group.
    returns:
        dict['ref_exp']
        dict['mexp_list']

    """
    # get each combo of metaexp
    combos = []
    for mexp in metaexp_list:
        if mexp.hasExp:
            combos.append((mexp.com + (mexp,)))
    res_list = [list(v) for l,v in groupby(sorted(combos, key=lambda x:(x[1], x[2])), lambda x: (x[1],x[2]))]

    final_groups = []
    for res in res_list:
        possible_days = [x[0] for x in res]
        com_string = [str(x[0])+'_'+str(x[1])+'_'+str(x[2])]

        possible_ind = [x for x,y in enumerate(possible_days) if y in day]

        # if there are very few choices for negative controls, select the last possible day.

        if len(possible_ind) < 1:
            possible_ind = [-1]

        rgroup = {}
        rgroup['ref_exp'] = res[possible_ind[0]][-1]
        rgroup['mexp_list'] = [x[-1] for x in res]
        rgroup['com_string'] = com_string
        final_groups.append(rgroup)
    return(final_groups)

## Goal is to calculate:
# total TNC produced per input cell from S1 and in S1 and S0
# total viable TNC produced per input cell from S1 and in S1 and S0
# total MKs per total S0 TNC input cell
# total MKs per total S0 CD34 input cell
# total viable MKs per total S0 TNC cell
# total viable MKs per total S0 CD34 cell
# Day by day fold change
# day by day fold change of MKs. number of MKs day 1/number of MKs day 0
# day by day fold change of viable MKs. number of MKs day 1/number of MKs day 0

# viability table
# exp, cb number, day 0, day 3, day
def lineplot(p):
    """
    todo: the background of plot is greyscale and the lines are gray instead of black. very annoying
    """

    plt.style.use('grayscale')
    plt.style.use('astroml')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p['x'], p['y'], color = 'k', linestyle='-', marker ='o', lw=2)
    ax.set_xlim(p['x_lim'])
    ax.set_xlabel(p['x_label'].replace('_',' '))
    ax.set_ylabel(p['y_label'].replace('_',' '))
    ax.set_title('{0!s}'.format(p['title']))
    fig.tight_layout()
    fig.savefig(p['fig_path'])

def get_production_perMK(cumul_prod_list, header, all_stats, s0_name = ['S0 TNC', 'S0 CD34', 'S0 CD34 CD90']):
    melted_list = []
    for cumul_prod in cumul_prod_list:
        melted = pd.melt(cumul_prod, header)
        melted['variable'].replace(regex=True, inplace=True, to_replace=r'\D',value=r'')
        melted.rename(columns = {'variable':'Day'}, inplace=True)
        melted_list.append(melted)
    current = melted_list[0].rename(columns={'variable':'Day', 'value': 'Cumul TNC per ' + s0_name[0]})
    for i, frame in enumerate(melted_list[1:], 1):
        current = current.merge(frame, on=header+['Day']).rename(columns={'variable':'Day','value': 'Cumul TNC per ' + s0_name[i]})
    current['Day'] = current['Day'].astype(int)
    combined = current.merge(all_stats, how='outer', on=header+['Day'])
    for s0 in s0_name:
        combined['Viable Cumul TNC per '+ s0] = combined['Cumul TNC per ' + s0] * combined['viability']
        combined['Viable Cumul 41p MK per ' + s0] =combined['Cumul TNC per '+s0]*combined['CD41p']*combined['viability']
        combined['Viable Cumul 41p42p MK per ' + s0] = combined['Cumul TNC per ' + s0] *combined['CD41p_CD42p']*combined['viability']
    return(combined)

def get_s0_fc(s0_col_name, s0_parsed, s1_fc_prod):
    df1 = s1_fc_prod.ix[:,3:]
    df2 = pd.concat([s0_parsed[s0_col_name]]*(s1_fc_prod.shape[1]-3), axis = 1)
    if s0_col_name == "S0 Fold Expansion VPA CD34+":
        df3 = pd.concat([100/s0_parsed['S0 Percentage VPA CD34+']]*(s1_fc_prod.shape[1]-3), axis = 1)
        total = pd.DataFrame(df1.values*df2.values*df3.values, columns=df1.columns, index=df1.index)
    elif s0_col_name == "S0 Fold Expansion VPA CD34+ CD90+":
        df3 = pd.concat([100/s0_parsed['S0 Percentage VPA CD34+ CD90+']]*(s1_fc_prod.shape[1]-3), axis = 1)
        total = pd.DataFrame(df1.values*df2.values*df3.values, columns=df1.columns, index=df1.index)
    else:
        total = pd.DataFrame(df1.values*df2.values, columns=df1.columns, index=df1.index)
    header = s1_fc_prod.ix[:,:3]
    return(pd.concat([header, total], axis=1))

def import_s0_expansion():
    df = pd.read_csv('data/s0_expansion_data.csv',encoding="utf-8-sig")
    df = df.dropna(axis=0,how='all')
    return(df)

def get_s0_results(raw_results, s0):
    header = raw_results.ix[:,:3]
    combined = header.merge(s0, how='left',on=['CB Number', 'PRE-EXPANSION DAY'])
    return(combined)

def import_s1_expansion(EXP_NUMBER):
    try:
        df = pd.read_csv('data/exp_{0!s}_s1_expansion.csv'.format(EXP_NUMBER), encoding="utf-8-sig", sep=',')
    except IOError:
        print("Exp {0!s} does not have expansion data.".format(EXP_NUMBER))
        return((None, None, None))
    if df.shape[1] == 1:
        df = pd.read_csv('data/exp_{0!s}_s1_expansion.csv'.format(EXP_NUMBER), encoding="utf-8-sig", sep='\t')

    df = df.dropna(axis=0,how='all')
    fc = df.ix[:,:3].copy()
    fc_prod = df.ix[:,:3].copy()

    running_fc = pd.Series(1, index=range(len(df)))
    col_name = df.ix[:,3].name
    if col_name == 'Day 0':
        fc['Day 0'] = pd.Series(1,index=range(len(df)))
        fc_prod['Day 0'] = pd.Series(1,index=range(len(df)))
    for i in range(3, len(df.columns), 2):
        col_name = df.ix[:,i+1].name
        fc[col_name] = df.ix[:,i+1]/df.ix[:,i]
        current_fc = df.ix[:,i+1]/df.ix[:,i]
        running_fc = running_fc.multiply(current_fc)
        fc_prod[col_name] = running_fc
        #if EXP_NUMBER == "18" :
        #    pdb.set_trace()
    return(df,fc, fc_prod)
# metaexp will parse table, get corresponding fc in S0 and S1

def parse_comp_sets(unique_voltages, tube_subset, volt_labels):
    """ Parses unique voltages and tube df to get a list of tube dicts for each tube

    Args:
        unique_voltages (dict): a dict with a list of voltages?
        tube_subset (df): a df with all surface and compensation tubes

    Return:

        dataframe with experiments

    """
    # For each voltage
    surface_list = []
    best_comp_path_list = []
    all_comp_path_list = []
    for volt in unique_voltages:
        target_voltage = dict(zip(volt_labels, volt))
        best_compensation_paths, all_compensation_paths = get_comp_tubes(target_voltage,tube_subset)
        best_comp_path_list.append(best_compensation_paths)
        all_comp_path_list.append(all_compensation_paths)

        surface_subset = tube_subset[tube_subset.isin(dict([ (x[0],[x[1]]) for x in target_voltage.items()]))[volt_labels].all(1)]
        surface_subset = surface_subset[surface_subset['TUBE TYPE'] == 'SURFACE']
        surface_list.append(surface_subset)


    e_list = []
    for tube_set in surface_list:
        if len(tube_set) > 0:
            tube_list = []
            for ind, row in tube_set.iterrows():
                ## experiments the following attributes
                ## Day (1-x)
                ## Expansion (5 or 7) - PRE-EXPANSION DAY
                ## Isotype (boolean)
                ## Treatment
                tube = flow.Tube(file = row['PATH'], conditions = {"Day":row['DAY'],
                                                                   "Expansion":row['PRE-EXPANSION DAY'],
                                                                   "Isotype":row['ISOTYPE'],
                                                                   "Treatment":row['TREATMENT'],
                                                                   "Replicate":row['REPLICATE'],
                                                                   "CB Number":row['CB Number'],
                                                                   "EXP": row['EXP']})
                tube_list.append(tube)

            import_op = flow.ImportOp(conditions = {'Day' : 'int',
                                                'Expansion' : 'category',
                                                'Isotype' : 'bool',
                                                'Treatment' : 'category',
                                                'Replicate' : 'int',
                                                'CB Number' : 'category',
                                                'EXP' : 'int'} , tubes = tube_list)
            exp_part = import_op.apply()
            e_list.append(exp_part)
            
    return(e_list, best_comp_path_list, all_comp_path_list)

def get_volts(tube_subset):
    """ Read pd df and identify unique volts

    Args:
        tube_subset (df): tube df

    Return:
        list of tuples? or a dict, I forgot

    """
    volt_labels = [x for x in tube_subset.columns if '-' in x]
    volt_labels = [x for x in volt_labels if 'PRE-EXPANSION DAY' not in x]
    all_voltages = tube_subset.groupby(volt_labels)
    unique_voltages = []
    for g,data in all_voltages:
        unique_voltages.append(g)
    return(unique_voltages, volt_labels)

def read_surface_tube_table(TUBE_FILE):
    """ Read a csv file with the preliminary assignments for each tube.

    Args:
        TUBE_FILE (str): tube path

    Returns:
        pandas dataframe with formatted surface tubes

    """

    expt_tubes = pd.read_csv(TUBE_FILE, sep = '\t')

    ### Convert columns into valid datatypes
    surface_tubes = expt_tubes[expt_tubes['TUBE TYPE'] == 'SURFACE']
    #surface_tubes[['DAY']] = surface_tubes[['DAY']].apply(pd.to_numeric)
    surface_tubes[['DAY']] = surface_tubes[['DAY']].convert_objects(convert_numeric=True)

    ### Assign additional statistics such as preliminary isotype pairing to each treatment file

    for ind, row in surface_tubes.iterrows():
        if row["ISOTYPE"] == False:
            result = surface_tubes[(surface_tubes["DAY"] == row["DAY"]) & (surface_tubes['PRE-EXPANSION DAY'] == row["PRE-EXPANSION DAY"]) & (surface_tubes['ISOTYPE'] == True)]
            if len(result) > 0:
                iso_path = result["PATH"].values[0]
            else:
                iso_path = "none"

            surface_tubes.loc[ind, 'ISO_FILE'] = iso_path
        else:
            surface_tubes.loc[ind, 'ISO_FILE'] = surface_tubes.loc[ind, 'PATH']

    return(surface_tubes, expt_tubes)


def generate_combos(iter_param, exp):
    """Generate combinations based on relevant parameters.

    Args:
        iter_param (dict) : param or column name (key) : datatype such as str, bool, int, etc (value)
        exp (exp object): experimental object

    Returns:
        a list of tuples representing possible combinations
    """
    all_items = []
    for param in sorted(iter_param.iterkeys()):
        items = np.sort(exp.data[param].dropna().unique())
        #items = [x for x in items if x != np.nan]
        all_items.append(items)

    # Generate all combinations
    combos = []
    for items in product(*all_items):
        combos.append(items)
    return(combos)

def gate_size(exp):
    """Gate-based on size (fixed, hard-coded values).
    Todo:
        add method for variable size gate
    Args:
        Exp (exp obj): experiment to be size-gated

    Returns:
        Experiment with size gated data
    """

    #2D range is:
    # xlow = 6504.61128019, xhigh = 260998.080801, ylow = 99.2728755467, yhigh=187986.103224
    r2d = flow.Range2DOp(name = "Size",
                         xchannel = "FSC-A",
                         ychannel = "SSC-A",
                         xlow = 1000.61128019,
                         xhigh = 260998.080801,
                         ylow = 99.2728755467,
                         yhigh=187986.103224)

    r2d.default_view(huefacet = "Treatment").plot(exp)
    size_exp = r2d.apply(exp)
    return(size_exp)

def get_fluormap(config='None'):
    fluormap = {}
    fluormap['CD34'] = "PE-A"
    fluormap['CD41'] = "FITC-A"
    fluormap['CD42'] = "APC-A"
    if config == 'exp_6':
        fluormap['CD34'] = "FITC-A"
        fluormap['CD41'] = "APC-A"
        fluormap['CD42'] = "PE-A"

    return(fluormap)


def get_iter_param():
    iter_param = {'Expansion':'str', 'Day':'int', 'Treatment':'str', 'CB Number':'str'}
    return(iter_param)

def plot_percent_comparison(color_table, line_map, treatment_color_map, timepoints = [0,3,5,7,9,11,13]):
    treatment_group =color_table.groupby(['Expansion', 'Treatment'])
    f, ax_tuple = plt.subplots(1, 4, figsize=(30,4))

    CD34_list = []
    CD41_list = []
    CD42_dp_list = []
    CD42_sp_list = []
    day_list = []

    for num,(name, group) in enumerate(treatment_group):
        if len(group) < 1:
            continue

        linestyle = line_map[name[0]]
        c = treatment_color_map[name[1]]

        group = group.sort('CD41+ CD42+').drop_duplicates(subset=['Day', 'Expansion', 'Treatment'], take_last=True).sort('Day')

        CD34=group['CD34+ CD41-'].values
        CD41=group['CD34- CD41+'].values
        CD42_dp=group['CD41+ CD42+'].values
        CD42_sp=group['CD41+ CD42-'].values

        day = group['Day'].values

        CD34_list.append(CD34)
        CD41_list.append(CD41)
        CD42_dp_list.append(CD42_dp)
        CD42_sp_list.append(CD42_sp)
        day_list.append(day)

        ax1 = ax_tuple[0]
        ax2 = ax_tuple[1]
        ax3 = ax_tuple[2]
        ax4 = ax_tuple[3]

        ax1.plot(day, [x*100 for x in CD34], linestyle = linestyle,marker='o', color = c, label = name)
        ax2.plot(day, [x*100 for x in CD41], linestyle = linestyle, marker = 'o', color = c, label = name)
        ax3.plot(day, [x*100 for x in CD42_dp], linestyle = linestyle, marker = 'o', color = c, label = name)
        ax4.plot(day, [x*100 for x in CD42_sp], linestyle = linestyle, marker = 'o', color = c, label = name)

        ax1.set_ylabel('% CD34 Positive Cells')
        ax2.set_ylabel('% CD41 Positive Cells')
        ax3.set_ylabel('% CD42 Positive/CD41 Positive Cells')
        ax4.set_ylabel('% CD42 Positive/CD41 Negative Cells')

        all_axes = [ax1, ax2, ax3, ax4]

        for ax in all_axes:
            ax.grid(False)
            ax.set_ylim([0,100])
            ax.set_xticks(timepoints)
            ax.set_xlabel('Day')
    legend = ax1.legend(loc='upper right', markerscale = 0.1)

def plot_percent_grid(color_table,timepoints = [0,3,5,7,9,11,13] ):
    treatment_group =color_table.groupby(['Expansion', 'Treatment'])

    f, ax_tuple = plt.subplots(len(treatment_group), 4, figsize=(30,4*(len(treatment_group))), squeeze=False)

    for num,(name, group) in enumerate(treatment_group):
        if len(group) < 1:
            continue

        group = group.sort('CD41+ CD42+').drop_duplicates(subset=['Day', 'Expansion', 'Treatment'], take_last=True).sort('Day')

        CD34=group['CD34+ CD41-'].values
        CD41=group['CD34- CD41+'].values
        CD42_dp=group['CD41+ CD42+'].values
        CD42_sp=group['CD41+ CD42-'].values

        day = group['Day'].values

        ax1 = ax_tuple[num][0]
        ax2 = ax_tuple[num][1]
        ax3 = ax_tuple[num][2]
        ax4 = ax_tuple[num][3]

        ax1.plot(day, [x*100 for x in CD34], 'ko-')
        ax2.plot(day, [x*100 for x in CD41], 'ko-')
        ax3.plot(day, [x*100 for x in CD42_dp], 'ko-')
        ax4.plot(day, [x*100 for x in CD42_sp], 'ko-')

        ax1.set_ylabel('% CD34 Positive Cells')
        ax2.set_ylabel('% CD41 Positive Cells')
        ax3.set_ylabel('% CD42 Positive/CD41 Positive Cells')
        ax4.set_ylabel('% CD42 Positive/CD41 Negative Cells')

        all_axes = [ax1, ax2, ax3, ax4]

        ax1.text(.5, .9, name[0] + ' ' + name[1] + ' | Live CD34+ Cell %', horizontalalignment = 'center', transform = ax1.transAxes)
        ax2.text(.5, .9, name[0] + ' ' + name[1] + ' | Live CD41+ Cell %', horizontalalignment = 'center', transform = ax2.transAxes)
        ax3.text(.5, .9, name[0] + ' ' + name[1] + ' | Live CD42+/CD41 Cell %', horizontalalignment = 'center', transform = ax3.transAxes)
        ax4.text(.5, .9, name[0] + ' ' + name[1] + ' | Live CD42+/CD41- Cell %', horizontalalignment = 'center', transform = ax4.transAxes)


        for ax in all_axes:
            ax.grid(False)
            ax.set_ylim([0,100])
            ax.set_xticks(timepoints)
            ax.set_xlabel('Day')

    f.subplots_adjust(hspace=.3)

def subset_by_condition(exp, condition_list):
    """
    Given an experiment and condition list, it divides the experiment
    into several different experiments.
    """
    result_list = []
    gps = exp.data.groupby(condition_list)
    gps_keys = gps.groups.keys()
    
    for key in gps_keys:
        result_dict = {}
        query = str('')
        for condition, value in zip(condition_list, key):
            result_dict[condition] = value
            query += '{} == "{}" and '.format(condition, value)
        
        query = query.rstrip(' and ')
        subexp= exp.query(query)        
        result_dict['exp'] = subexp
        result_list.append(result_dict)
    return(result_list)

def get_compensation_gates(compensation_paths):
    tube_list = []
    for color in compensation_paths.keys():
        tube = flow.Tube(file = compensation_paths[color], conditions = {"Color":color})
        tube_list.append(tube)

    import_op = flow.ImportOp(conditions = {'Color' : 'category'},tubes = tube_list)
    comp_ex = import_op.apply()

    g = flow.GaussianMixture2DOp(name = "Debris_Filter",
                             xchannel = "FSC-A",
                             xscale = "logicle",
                             ychannel = "SSC-A",
                             yscale = "logicle",
                             num_components = 4,
                             sigma = 2)
    g.estimate(comp_ex)
    bead_coords = g.default_view().plot(comp_ex,get_coords = 3)
    gated_comp = g.apply(comp_ex)
    gated_comp.data
    color = 'PE-A'

    my_subset = "Color == '" + color +"' and Debris_Filter == 'Debris_Filter_3'"
    flow.HistogramView(channel = color, subset = my_subset).plot(gated_comp)

    bead_coords = [tuple(l) for l in bead_coords]

    # Create polygon gate out of gaussian gate
    bead_gate = flow.PolygonOp(name = "Bead", xchannel = "FSC-A", ychannel = "SSC-A", vertices=bead_coords)
    pgated_comp = bead_gate.apply(comp_ex)

    bl_op = flow.BleedthroughLinearOp()
    bl_op.controls = compensation_paths

    bl_op.estimate(pgated_comp, subset = "Bead == True")
    bl_op.default_view().plot(pgated_comp)
    return(bl_op, pgated_comp, bead_coords)


def get_isotype_gates(gated_subset):
    ## Get the isotype gate.
    g_pe_fitc = flow.GaussianMixture2DOp(name = "PE-FITC-GM",
                         xchannel = "PE-A",
                         xscale = "logicle",
                         ychannel = "FITC-A",
                         yscale = "logicle",
                         num_components = 1,
                         sigma = 5)
    g_pe_fitc.estimate(gated_subset)
    pe_fitc_iso_gate = g_pe_fitc.default_view().plot(gated_subset,get_coords = 'bl')
    pf_right_corner = max(pe_fitc_iso_gate,key=lambda item:item[0]**2+item[1]**2)
    qg_pf =flow.QuadOp(name="PE-FITC", xchannel = "PE-A", ychannel = "FITC-A", xthreshold = pf_right_corner[0], ythreshold = pf_right_corner[1])

    g_fitc_apc = flow.GaussianMixture2DOp(name = "FITC-APC-GM",
                         xchannel = "FITC-A",
                         xscale = "logicle",
                         ychannel = "APC-A",
                         yscale = "logicle",
                         num_components = 1,
                         sigma = 4)
    g_fitc_apc.estimate(gated_subset)
    fitc_apc_iso_gate = g_fitc_apc.default_view().plot(gated_subset,get_coords = 'bl')
    fa_right_corner = max(fitc_apc_iso_gate,key=lambda item:item[0]**2+item[1]**2)
    qg_fa =flow.QuadOp(name="FITC-APC", xchannel = "FITC-A", ychannel = "APC-A", xthreshold = fa_right_corner[0], ythreshold = fa_right_corner[1])

    return(qg_pf, qg_fa)

def get_query(com, iter_param):
    query_str = ""
    for ix,param in enumerate(sorted(iter_param.iterkeys())):
        parsed_param = str(param).replace(" ","_")
        if iter_param[param] == 'str':
            query_str = query_str + str(parsed_param) + ' == ' + "\"" +str(com[ix])+ "\"" + ' and '
        elif iter_param[param] == 'int':
            query_str = query_str + str(parsed_param) + ' == ' +str(com[ix]) + ' and '
    query_str = re.sub(' and $', '', query_str)
    return(query_str)

def convert_coords(coords):
    # takes a list of lists, converts to list of tuples
    return([tuple(l) for l in coords])

def get_logicle(my_exp,channels, log=False):
    if not log:
        logicle = flow.LogicleTransformOp()
        logicle.channels = channels
        logicle.estimate(my_exp)
    else:
        logicle = flow.LogTransformOp()
    return(logicle.apply(my_exp))

def get_2d_mask(xaxis,yaxis,axis_list, mat):
    x_idx = axis_list.index(xaxis)
    y_idx = axis_list.index(yaxis)
    mask = np.array([x_idx,y_idx])

    mat_2d = mat[mask[:,None], mask]
    return(mat_2d)

def get_gated_comp(comp_ex, color, plt = True, sigma = 1):
    query_str = "Color == '" + color + "'"
    color_comp = comp_ex.query(query_str)
    g = flow.GaussianMixture2DOp(name = "Debris_Filter",
                                 xchannel = "FSC-A",
                                 xscale = "logicle",
                                 ychannel = "SSC-A",
                                 yscale = "logicle",
                                 num_components = 5,
                                 sigma = sigma)
    g.estimate(color_comp)
    if plt:
        g.default_view().plot(color_comp)
    gated_comp = g.apply(color_comp)
    return(gated_comp)

def plot_compensation_grid(comp_ex, corrected_comp_ex, color, channels, log=False):

    f, axarr = plt.subplots(3)
    #gated_comp = get_gated_comp(comp_ex, color, plt=False)
    #corrected_gated_comp = get_gated_comp(corrected_comp_ex, color, plt=False)
    my_subset = "Color == '" + color +"' and Bead == True"
    gated_comp = comp_ex.query(my_subset)

    corrected_gated_comp = corrected_comp_ex.query(my_subset)
    gated_comp = get_logicle(gated_comp,channels, log = log)
    corrected_gated_comp = get_logicle(corrected_gated_comp,channels, log = True)

    plotted = []
    i = 0
    for from_idx, from_channel in enumerate(channels):
        for to_idx, to_channel in enumerate(channels):
            if (from_idx == to_idx) or ((to_idx,from_idx) in plotted):
                continue
            # Three subplots sharing both x/y axes
            xc = channels[from_idx]
            yc = channels[to_idx]
            x,y,z = get_channel_data(gated_comp, xc, yc)
            axarr[i].scatter(x, y, c=z, edgecolor = '', antialiased=True, alpha = 0.4, s=2, marker='o', cmap=plt.get_cmap('Blues'))
            axarr[i].set_xlabel(xc, fontsize=12)
            axarr[i].set_ylabel(yc, fontsize=12)
            x2,y2,z2 = get_channel_data(corrected_gated_comp, xc, yc)
            axarr[i].scatter(x2, y2, c=z2, edgecolor = '', antialiased=True, alpha = 0.4, s=2, marker='o', cmap=plt.get_cmap('Spectral_r'))
            # Fine-tune figure; make subplots close to each other and hide x ticks for
            # all but bottom plot.
            plotted.append((from_idx, to_idx))
            i += 1
    f.subplots_adjust(hspace=0.5)
    #plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=True)
    f.suptitle(color + " Beads", fontsize=14)


def parse_tube_type(parentdir, meta):
    bead_names = ['beads', 'apc', 'fitc', 'pe','bv']

    if 'ploidy' in parentdir.lower():
        tube_type = 'PLOIDY'
    elif 'platelet' in parentdir.lower():
        tube_type = 'PLATELET'
        day = 'NA'
        vpa_day = 'NA'
        treatment = 'NA'
    elif any(x in meta['TUBE NAME'].lower() for x in bead_names) and 'iso' not in meta['TUBE NAME'].lower():
        tube_type = 'BEADS'
        day = 'NA'
        vpa_day = 'NA'
        treatment = 'NA'
    else:
        tube_type = 'SURFACE'
    
    return(tube_type)

def get_tube_info(path, EXP_NUMBER = 16, DEFAULT_CB='nan'):

    parentdir = path.split('/')[-2]

    tube_info = [('PARENT FOLDER',parentdir)]
    meta = fcsparser.parse(path, meta_data_only=True, reformat_meta=True)
    
    # Get color/channel dictionary
    color_channel = dict(zip(meta['_channels_']['$PnN'], meta['_channels_'].index.tolist()))
    color_channel = {color:"$P{}V".format(channel) for color, channel in color_channel.items()}
    #color_channel = pd.DataFrame(color_channel, columns=['Channel Name', 'Current Channel Number'])

    # Get channel/volt dictionary
    channel_volt={k:v for k,v in meta.items() if k.startswith('$P')}

    # Add color/volt dictionary to the tube_info
    color_volt = {color:channel_volt[channel] for color, channel in color_channel.items() if color != 'Time'}
    tube_info.extend(color_volt.items())

    # Add some selected parameters from the metadata to the tube_info
    target_params = ['TUBE NAME', 'EXPERIMENT NAME', '$DATE' ]
    param_data = [meta[target_params[x]] for x,param in enumerate(target_params)]
    tube_info.extend(zip(target_params,param_data))

    # Parse the tube type
    tube_type = parse_tube_type(parentdir, meta)

    # Parse tube pre-expansion
    if tube_type == 'SURFACE' or tube_type == 'PLOIDY':
        meta['TUBE NAME'] = path.split('_')[-1].rstrip('.fcs')

        if 'e7' in meta['TUBE NAME'].lower():
            vpa_day = 'E7'
        elif 'e5' in meta['TUBE NAME'].lower():
            vpa_day = 'E5'
        elif 'e0' in meta['TUBE NAME'].lower():
            vpa_day = 'E0'
        elif 'e7' in parentdir.lower():
            vpa_day = 'E7'
        elif 'e5' in parentdir.lower():
            vpa_day = 'E5'
        elif 'e0' in parentdir.lower():
            vpa_day = 'E0'

        # determine current day
        p = re.compile(r''+ vpa_day + ' Day (\d+)')
        print(parentdir)
        match = p.search(parentdir)
        day = match.group(1)
    else:
        vpa_day = 'nan'
        day = 'nan'
    # parse treatment from the tube name
    treatment = meta['TUBE NAME']

    if 'iso' in meta['TUBE NAME'].lower():
        iso_bool = True
    else:
        iso_bool = False

    CB_regex = re.compile('CB([0-9]*)')
    CB_number = CB_regex.findall(meta['TUBE NAME'])
    if len(CB_number) == 0:
        CB_number = DEFAULT_CB
    else:
        CB_number = 'CB'+CB_number[0]

    tube_info.extend([('TUBE TYPE',tube_type)])
    tube_info.extend([('TREATMENT',treatment)])

    tube_info.extend([('DAY',day)])
    tube_info.extend([('PRE-EXPANSION DAY',vpa_day)])
    tube_info.extend([('ISOTYPE', iso_bool)])
    tube_info.extend([('PATH', path)])
    tube_info.extend([('REPLICATE', 1)])
    tube_info.extend([('SRC',meta['$SRC'])])
    tube_info.extend([('EXP', EXP_NUMBER )])
    tube_info.extend([('CB Number', CB_number )])
    tube_info = pd.Series(dict(tube_info))

    return(tube_info)

def get_comp_tubes(target_voltage,expt_tubes):
    compensation = expt_tubes[expt_tubes['TUBE TYPE'] == 'BEADS']
    color_names = ['apc','fitc','pe']
    converted_color_names = ['APC-A','FITC-A','PE-A']
    compensation[converted_color_names] = compensation[converted_color_names].convert_objects(convert_numeric=True)
    #compensation[voltage_names] = compensation[voltage_names].apply(pd.to_numeric)
    compensation[['$DATE']] = compensation[['$DATE']].apply(pd.to_datetime)
    
    best_comp_tubes = pd.DataFrame()
    same_voltage = pd.DataFrame()

    for x,current_color in enumerate(color_names):
        comp_subset = compensation[(pd.DataFrame([compensation[k] == v for k, v in target_voltage.iteritems()]).all()) & (compensation['TUBE NAME'].str.contains(current_color, case = False))]
        same_voltage = same_voltage.append(comp_subset)

        if 'BESTCOMP' in comp_subset.columns and len(comp_subset[comp_subset['BESTCOMP']==True]) > 0:
            best_tube = comp_subset[comp_subset['BESTCOMP']==True].iloc[0]
        else:
            best_tube = comp_subset.sort_values('$DATE').min()
        
        best_comp_tubes = best_comp_tubes.append(best_tube, ignore_index=True)
    
    return(best_comp_tubes, same_voltage)

def get_merged_exp(exp_list):
    new_exp = exp_list[0].clone()
    exp_list.pop(0)
    for exp in exp_list:
        new_exp.merge_events(exp.data)
    return(new_exp)

def get_merged_data(data_list):
    new_data = data_list[0].copy()
    data_list.pop(0)
    for data in data_list:
        new_data = new_data.append(data, ignore_index = True)
    return(new_data)
