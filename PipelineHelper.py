import pickle
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from CompensationHandler import CompensationHandler
from CompensationReport import CompensationReport
from FlowCollection import FlowCollection
import utility as util
import pdb
import os

def compensate(expt_tubes, settings, perform=True, report=True, title=""):

    tube_subset = expt_tubes[(expt_tubes['TUBE TYPE'] == 'BEADS') | (expt_tubes['TUBE TYPE'] == 'SURFACE')]
    # get the voltage settings for each experiment
    # if an experiment has more than one voltage settings, stratify the experiment based on voltage setting
    unique_voltages, volt_labels = util.get_volts(tube_subset)
    e_list, best_comp_path_list, all_comp_path_list = util.parse_comp_sets(unique_voltages, tube_subset, volt_labels)

    corrected_list = []
    #todo: move parse_comp_sets into CompensationHandler
    if settings.get('COMPENSATION', True):

        for n,volts in enumerate(unique_voltages):

            experiment = e_list[n]
            best_comp = best_comp_path_list[n]
            all_comp = all_comp_path_list[n]
            volt_dict = dict(zip(volt_labels, volts))
            ch = CompensationHandler(experiment, volt_dict, all_comp, best_comp)
            
            corrected_list.append(ch.comp_exp)

            # Getting the compensation report for each voltage set
            section_title = "Voltage Set {}".format(n)
            creport = CompensationReport(title)
            creport.add_comp_graphs(ch, title=section_title)

            creport_directory = "reports/{}/compensation_reports".format(settings['EXP_NAME'])
            creport_savepath = "{}/creport_voltage_{}.png".format(creport_directory,n)
            
            # Create a folder if it does not exist
            if not os.path.exists(creport_directory):
                try:
                    os.makedirs(creport_directory)
                except OSERROR as e:
                    if e.errno != errno.EEXIST:
                        raise
            creport.render(savepath = creport_savepath)
            
    else:
        for ind,e in enumerate(e_list):
            corrected_list.append(e)

    # merge the voltage-grouped, compensation-corrected experiments into one experiment
    exp_comp = util.get_merged_exp(corrected_list)
    uncorrected_list = []
    for ind,e in enumerate(e_list):
        uncorrected_list.append(e)

    exp = util.get_merged_exp(uncorrected_list)

    compensation = {    'compensated': exp_comp,
                        'uncompensated': exp}

    return(compensation)

def get_groups(exp, groupby):
    return(exp.data.groupby(groupby).groups.keys())

def aggregateData(fc_list):
    result_df = pd.DataFrame()

    for fc in fc_list:
        fc_result = dict()
        # for each FC, get the between population features
        fc_result['index']= (fc.grouping[1],fc.grouping[0], fc.grouping[2])
        fc_result.update(fc.between_pop)

        for mexp in fc.metaexps:
            # for each population, get the population name
            for pop in mexp.populations:
                pop_dict = pop.pop_features
                cluster_ind = pop_dict['cluster_idx']
                new_pop_dict = {'Pop{0!s}.Day{1!s}.{2!s}'.format(cluster_ind,pop.day,k): v for k,v in pop_dict.items() if k != 'cluster_idx'}
                fc_result.update(new_pop_dict)
        
        culture_result = pd.Series(fc_result)
        result_df = result_df.append(culture_result,ignore_index=True)
    
    return(result_df)

def QC(expt_tubes, settings, perform=True, report=True):
    title = "Compensation Report {}".format(settings['EXP_NAME'])
    compensation = compensate(expt_tubes, settings, perform=perform, report=report, title=title)
    exp_comp = compensation['compensated']
    exp_uncomp = compensation['uncompensated']
    return(exp_comp, exp_uncomp)

def executeXDGating(settings, save=True, load=False):
    if not load:
        surface_tubes, expt_tubes = util.read_surface_tube_table(settings['TUBE_FILE'])
        exp_comp, exp_uncomp = QC(expt_tubes, settings, perform = True, report = True)
        iter_param = {'Expansion':'str', 'Day':'int', 'Treatment':'str', 'CB Number':'str'}

        # Get the FlowCollections of each condition
        groups = get_groups(exp_comp, ['CB Number','Expansion', 'Treatment'])

        fc_list = []
        for gr in groups:
            fc_list.append(FlowCollection(exp_comp, iter_param, gr, settings))   

        for fc in fc_list:
            fc.gateXD(settings) 
            #fc.plot_3D('PE-A', 'FITC-A', 'APC-A')
    else:
        with open('pickles/controls_only/surface_tubes.pkl', 'rb') as handle:
            surface_tubes = pickle.load(handle)
        with open('pickles/controls_only/expt_tubes.pkl', 'rb') as handle:
            expt_tubes = pickle.load(handle)
        with open('pickles/controls_only/exp_comp.pkl', 'rb') as handle:
            exp_comp = pickle.load(handle)
        with open('pickles/controls_only/exp_uncomp.pkl', 'rb') as handle:
            exp_uncomp = pickle.load(handle)
        with open('pickles/controls_only/fc_list.pkl', 'rb') as handle:
                    fc_list = pickle.load(handle)
    
    if save:
        with open('pickles/controls_only/surface_tubes.pkl', 'wb') as handle:
            pickle.dump(surface_tubes, handle, protocol = pickle.HIGHEST_PROTOCOL)

        with open('pickles/controls_only/expt_tubes.pkl', 'wb') as handle:
            pickle.dump(expt_tubes, handle, protocol = pickle.HIGHEST_PROTOCOL)

        with open('pickles/controls_only/exp_comp.pkl', 'wb') as handle:
            pickle.dump(exp_comp, handle, protocol = pickle.HIGHEST_PROTOCOL)

        with open('pickles/controls_only/exp_uncomp.pkl', 'wb') as handle:
            pickle.dump(exp_uncomp, handle, protocol = pickle.HIGHEST_PROTOCOL)

        with open('pickles/controls_only/fc_list.pkl', 'wb') as handle:
            pickle.dump(fc_list, handle, protocol = pickle.HIGHEST_PROTOCOL)

    all_pop_df = pd.DataFrame()
    current_index = 0
    for fc in fc_list:
        fc_name = fc.grouping
        for mexp in fc.metaexps:
            mexp_name = mexp.day
            for pop in mexp.populations:
                pop_info = {    'idx': current_index,
                                'mexp_name': mexp_name,
                                'fc_name': fc_name}
                pop_info.update(pop.scaled_means)

                all_pop_df = all_pop_df.append( pop_info, ignore_index=True)
                current_index += 1

    col_names = ['idx','mexp_name','fc_name'] + pop.axis_list
    all_pop_df = all_pop_df[col_names]
    #distance_matrix = pd.DataFrame(squareform(pdist(all_pop_df.iloc[:, 3:])), columns=all_pop_df.idx.unique(), index=all_pop_df.idx.unique())
    #linkage = hc.linkage(distance_matrix, method='average')
    clust_map = sns.clustermap(all_pop_df.iloc[:,3:], row_cluster=True, col_cluster=False)
    dendrogram = clust_map.dendrogram_row.dendrogram.keys()
    linkage = clust_map.dendrogram_row.linkage
    #fancy_dendrogram(linkage,truncate_mode='lastp',p=12,leaf_rotation=90.,leaf_font_size=12.,show_contracted=True,annotate_above=10)
    # 405, 4

    #around .2 works well
    max_d = .3
    clusters = fcluster(linkage, max_d, criterion='distance')
    all_pop_df['cluster_membership'] = clusters
    all_pop_df.to_csv('max_d_point3.csv')
    unique_clusters=all_pop_df['cluster_membership'].unique()
    #_plot_clust_scatter(all_pop_df,'FSC-A','DAPI-A')
    #_plot_clust_scatter(all_pop_df,'PE-A','FITC-A')
    #_plot_clust_scatter(all_pop_df,'FITC-A','APC-A')

    current_index = 0
    for fc in fc_list:
        fc_name = fc.grouping
        for mexp in fc.metaexps:
            mexp_name = mexp.day
            for pop in mexp.populations:
                pop.cluster_idx = all_pop_df['cluster_membership'][current_index]
                current_index += 1
                
    for fc in fc_list:
        fc.generateBetweenPopulationFeatures(clusters)

    result_df=aggregateData(fc_list)
    controls_table = pd.read_csv('data/new_agg_df_01112018.csv')
    from ast import literal_eval
    controls_table['index'] = [literal_eval(x) for x in controls_table['index']]
    merged_tables = result_df.merge(controls_table, on='index')
    valid_columns = merged_tables.count()[merged_tables.count() > 5].index
    merged_tables = merged_tables.fillna(0)
    merged_tables[valid_columns].to_csv('partial_controls_data.csv', index=False)
    pdb.set_trace()
    return(fc_list)