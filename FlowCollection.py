from MetaExp import MetaExp
import utility as util
import cytoflow as flow
import math
import pandas as pd

import pdb

class FlowCollection:

    def __init__(self, exp, iter_param, grouping, settings):
        
        self.metaexps = []
        self.days = []
        self.grouping = grouping

        subset_iter_param = {'Expansion':'str', 'Treatment':'str', 'CB Number': 'str'}
        query = util.get_query(grouping, subset_iter_param)
        sub_exp = exp.query(query)
        self.days = sorted(sub_exp.data['Day'].unique())
         
        # get all the days in this group
        # order the days
        for day in self.days:
            com = list(grouping)
            com.insert(1,day)
            metaexp = MetaExp(exp, com, iter_param, settings=settings)
            self.metaexps.append(metaexp)

        # create a list of MetaExp

        self.channel_dict = {   'CD34': self.metaexps[0].CD34_name,
                                'CD41': self.metaexps[0].CD41_name,
                                'CD42': self.metaexps[0].CD42_name}
    
    def check_channels(self):
        """ 
        Checks if the channel/marker names are consistent within a certain flow collection.
        Prints a warning if inconsistencies are detected. 
        """
        cd_34 = []
        cd_41 = []
        cd_42 = []

        for metaexp in self.metaexps:
            cd_34.append(metaexp.CD34_name)
            cd_41.append(metaexp.CD41_name)
            cd_42.append(metaexp.CD42_name)

        markers = [cd_34, cd_41, cd_42]

        for marker in markers:
            all_same = marker.count(marker[0]) == len(marker)

            if not all_same:
                print("Warning: Marker/Channel is not consistent within Flow Collection {}".format(self.grouping))
    
    def plot_3D(self, xaxis, yaxis, zaxis):
        for metaexp in self.metaexps:
            metaexp.plot_3D(xaxis, yaxis, zaxis)

    def gateXD(self, settings):
        self.max_est_components = 10

        for metaexp in self.metaexps:
            metaexp.estimateXD(settings, max_ncomp = self.max_est_components)
            metaexp.gateXD(settings)
            metaexp.generatePopulations()
            #metaexp.plot_2D()
    
    def generateBetweenPopulationFeatures(self, clusters):
        days = self.days

        day_combo = zip(days, days[1:])
        # change in 7 over time
        self.between_pop = {}
        for clust in clusters:
            for day_a,day_b in day_combo:
                print(day_a, day_b)
                delta_name = 'clust_{0!s}_flux_{1!s}to{2!s}'.format(clust, day_a, day_b)
                pops_day_a = [pop for meta in self.metaexps for pop in meta.populations if (meta.day==day_a) & (len(pop.df)>0) & (pop.cluster_idx == clust)]
                pops_day_b = [pop for meta in self.metaexps for pop in meta.populations if (meta.day==day_b) & (len(pop.df)>0) & (pop.cluster_idx == clust)]
                if (len(pops_day_a) == 0) and (len(pops_day_b) == 0):
                    self.between_pop[delta_name] = 0
                elif (len(pops_day_a) == 0) and (len(pops_day_b) > 0):
                    self.between_pop[delta_name] = pops_day_b[0].pop_features['percent_membership']
                elif (len(pops_day_a) > 0) and (len(pops_day_b) == 0):
                    self.between_pop[delta_name] = -pops_day_a[0].pop_features['percent_membership']
                else:  
                    self.between_pop[delta_name]=pops_day_b[0].pop_features['percent_membership']-pops_day_a[0].pop_features['percent_membership']

                pops_day_a1 = [pop for meta in self.metaexps for pop in meta.populations if (meta.day==day_a) & (len(pop.df)>0)]
                pops_day_b1 = [pop for meta in self.metaexps for pop in meta.populations if (meta.day==day_b) & (len(pop.df)>0)]

                most_a1 = sorted(pops_day_a1, key=lambda x: x.scaled_means['FITC-A'], reverse=True)
                most_b1 = sorted(pops_day_b1, key=lambda x: x.scaled_means['FITC-A'], reverse=True)

                #Euclidean distance between the means of the most CD41 population.
                percent_change_name = 'mean_mfi_rate_41_{0!s}to{1!s}'.format(day_a,day_b)
                self.between_pop[percent_change_name]= math.sqrt(sum((most_b1[0].scaled_means[k] - most_a1[0].scaled_means[k])**2 for k in most_b1[0].scaled_means.keys()))
        return(self.between_pop)

    def get_table(self):
        my_df = pd.DataFrame(self.between_pop)
        
    def gateLiveDead(self, augmented=True):
        for metaexp in self.metaexps:
            metaexp.gateLiveDead(augmented=augmented)
            metaexp.calcViability()

    def get_live_bool_data(self):
        live_bool = util.get_merged_exp([x.exp_live_bool for x in self.metaexps if x.hasExp])
        return(live_bool)
    
    def get_live_dead_data(self):
        live_dead = util.get_merged_exp([x.exp_live_dead for x in self.metaexps if x.hasExp])
        return(live_dead)
    
    def get_live_data(self):
        live = util.get_merged_exp([x.exp_live for x in self.metaexps if x.hasExp])
        return(live)

    def get_dead_data(self):
        dead = util.get_merged_exp([x.exp_dead for x in self.metaexps if x.hasExp])
        return(dead)