import math
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import chi2, norm
from sklearn import mixture

import cytoflow as flow
import cytoflow.utility as cytoutil
import utility as util
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from Report import Single3DFigure, Single2DFigure

from Population import Population
import re


class MetaExp:

    def __init__(self, exp, combination, iter_param, settings=0, index=0):
        """
        Parameters
        ----------
            exp (exp obj):
            combination (str):
            iter_param (dict): dict with param name (key): type (value)

        Return
        ------
            none

        """
        self.com = combination
        self.query = util.get_query(combination, iter_param)
        self.exp = exp.query(self.query)
        self.settings = settings
        self.day = combination[1]

        self.index = index
        if not len(self.exp.data):
            self.hasExp = False
        else:
            self.hasExp = True
            #self.EXP_NAME = "{!s}".format(self.exp.data['EXP'].iloc[0])
            self.EXP_NAME = self.settings['EXP_NAME']
            self.cb_number = "{!s}".format(self.exp.data['CB Number'].iloc[0])
            if self.EXP_NAME == '6':
                self.CD34_name = "FITC-A"
                self.CD41_name = "APC-A"
                self.CD42_name = "PE-A"
            else:
                self.CD34_name = "PE-A"
                self.CD41_name = "FITC-A"
                self.CD42_name = "APC-A"

        self.exp_live_dead = None
        self.live_gate = None
        self.dead_gate = None
        self.surface_gates = {}
        self.surface_gate_coords = {}
        self.stats = {}
        self.populations = []
        self.populations_summary = pd.DataFrame()

        self.gate_stats = {}

    def _gateLiveCells(self, size, augmented=True):

        exp_subset = self.exp

        gates = []
        if size:
            if "PAC-A" in exp_subset.channels:
                blue_channel = "PAC-A"

            dapi_thresh = flow.ThresholdOp(name = "DAPI_Live", channel= blue_channel, threshold=0)
            live_thresh = flow.ThresholdOp(name = "FSC_Live", channel = "FSC-A", threshold = 1000)
            gates.append(dapi_thresh)
            gates.append(live_thresh)

            exp_subset = dapi_thresh.apply(self.exp)
            exp_subset = live_thresh.apply(exp_subset)

        ### Gaussian gate estimation and conversion to polygon gate
        g1 = flow.GaussianMixture2DOp(  name = "Live",
                                        xchannel = "FSC-A",
                                        xscale = "logicle",
                                        ychannel = blue_channel,
                                        yscale = "logicle",
                                        num_components = 2,
                                        sigma = 4)
        if size:
            g1.estimate(exp_subset, subset = "FSC_Live == True and DAPI_Live == True")
        else:
            g1.estimate(exp_subset)
        g_results = g1.default_view().plot(exp_subset, get_coords = 'br', get_stats=True)
        
        
        self.live_vertices = util.convert_coords(g_results[0])
        self.live_mean = g_results[1]
        self.live_covar = g_results[2]
        self.live_xl = g_results[3]
        self.live_yl = g_results[4]
        
        if augmented:
            aug_vert, aug_gate =self.augmentGate(self.live_vertices, exp_subset, blue_channel, gate_name = "Live", xchannel="FSC-A")
            self.live_vertices = aug_vert
            self.live_gate = aug_gate
        else:
            self.live_gate = flow.PolygonOp(name = 'Live', xchannel = 'FSC-A', ychannel = blue_channel, vertices = self.live_vertices)
        gates.append(self.live_gate)

        self.exp_live = self.exp.clone()
        for gate in gates:
            self.exp_live = gate.apply(self.exp_live)
        self.exp_live_bool = self.exp_live.clone()
        plt.close('all')
    
    def augmentGate(self, verts, exp_subset, blue_channel, gate_name, xchannel):
        """ Returns the augmented polygon vertices and gate"""
        polygons = []

        polygons.append(Polygon(verts))

        xscale = flow.utility.scale_factory('logicle', exp_subset, xchannel)
        yscale = flow.utility.scale_factory('logicle', exp_subset, blue_channel)
        polygons.append(Polygon(self.getAugmentedVertices(verts, xscale, yscale)))
        
        augmented_live = cascaded_union(polygons)
        aug_vertices = zip(*augmented_live.exterior.coords.xy)
        aug_gate = flow.PolygonOp(name = gate_name, xchannel = xchannel, ychannel = blue_channel, vertices = aug_vertices)
        
        return(aug_vertices, aug_gate)
    
    def getAugmentedVertices(self,verts, xscale, yscale):

        x_verts = [i[0] for i in verts]
        y_verts = [i[1] for i in verts]

        x_verts = xscale(np.array(x_verts))
        y_verts = yscale(np.array(y_verts))

        half_point_x=(min(x_verts) + max(x_verts))/2
        half_point_y=(min(y_verts) + max(y_verts))/2
        
        length_x = max(x_verts) - min(x_verts)
        length_y = max(y_verts) - min(y_verts)
        # bottom left corner
        bl = (xscale.inverse(half_point_x-.05*length_x), yscale.inverse(half_point_y))
        br = (xscale.inverse(half_point_x + length_x/2 + length_x*.2), yscale.inverse(half_point_y))
        tl = (xscale.inverse(half_point_x-.05*length_x), yscale.inverse(half_point_y + length_y/2 + length_y*.35))
        tr = (xscale.inverse(half_point_x + length_x/2 + length_x*.5), yscale.inverse(half_point_y + length_y/2 + 0.7*length_y))

        shape = [bl, tl, tr, br]
        return(shape)

    def _gateDeadCells(self):

        gates = []
        exp_subset = self.exp.clone()
        blue_channel ="DAPI-A"
        if "PAC-A" in exp_subset.channels:
            blue_channel = "PAC-A"
        thresh = flow.RangeOp(name = "DAPI", channel = blue_channel, low = 60000, high = 250000)
        dead_gate1 = thresh.apply(exp_subset)
        
        gates.append(thresh)

        g2 = flow.GaussianMixture2DOp(  name = "Dead",
                                        xchannel = "FSC-A",
                                        xscale = "logicle",
                                        ychannel = blue_channel,
                                        yscale = "logicle",
                                        num_components = 1,
                                        sigma = 4)
        g2.estimate(dead_gate1, subset = "DAPI == True")

        d_results = g2.default_view().plot(dead_gate1, get_coords = 1, get_stats=True)
        self.dead_vertices = util.convert_coords(d_results[0])
        self.dead_mean = d_results[1]
        self.dead_covar = d_results[2]
        self.dead_xl = d_results[3]
        self.dead_yl = d_results[4]

        self.dead_gate = flow.PolygonOp(name = "Dead", xchannel = "FSC-A", ychannel = blue_channel, vertices = self.dead_vertices)
        
        aug_vert, aug_gate =self.augmentGate(self.dead_vertices, exp_subset, blue_channel, gate_name = "Dead", xchannel="FSC-A")
        self.dead_vertices = aug_vert
        self.dead_gate = aug_gate

        gates.append(self.dead_gate)

        self.exp_dead = self.exp.clone()
        for gate in gates:
            self.exp_dead = gate.apply(self.exp_dead)
        
        plt.close('all')

    def calcViability(self):
        exp_live = self.exp_live.query("Live == True and Isotype == False ")
        exp_dead = self.exp_dead.query("Dead == True and Isotype == False ")
        self.stats['viability'] = (float(len(exp_live))/float((len(exp_live) + len(exp_dead))))

    #def calcSurface(self, channel_name

    def extractPopulations2D(self, names, boundaries):

        gate_type = 'gauss'
        xchannel = 'PE-A'
        ychannel = 'FITC-A'

        boundaries = {  'CD34': 300,
                        'CD41': 100,
                        'CD42': 200}

        color2cd = {'PE-A' : 'CD34',
                    'FITC-A': 'CD41',
                    'APC-A': 'CD42' }


        cd2color = {v: k for k, v in color2cd.iteritems()}

        # Get all populations in 1) CD34_CD41 dimension, CD41_CD42 dimension

        #pop_list = get_pop_stats()
        # 3 populations
        # 6 statistics per population
        # classify each population
        #self.stats[
        # xchannel_ychannel__popindex__xchannel_mean
        # xchannel_ychannel__popindex__ychannel_mean
        # xchannel_ychannel__popindex__xchannel_covlength
        # xchannel_ychannel__popindex__ychannel_covlength
        # xchannel_ychannel__popindex__eccentricity
        # xchannel_ychannel__popindex__percentevents

        # classification of each populationbased on means and names and boundaries
        name_bank = ('CD34_low__CD41_low', 'CD34_low__CD41_high', 'CD34_high__CD41_low','CD34_high__CD41_high', 'CD41_low__CD42_low', 'CD41_low__CD42_high', 'CD41_high__CD42_low','CD41_high__CD42_high')

        used_names = []
        pop_index = 1
        # if name exists, increment last number
        pass

    def _makeXDGMM(self, data, n_components, axis_list, scales):
        x = data.loc[:,axis_list]
        for idx,scale in enumerate(scales):
            x[axis_list[idx]] = scale(x[axis_list[idx]])
            x = x[~(np.isnan(x[axis_list[idx]]))]

        x = x.values
        #gmm = mixture.GaussianMixture(n_components = n_components, covariance_type = "full", random_state = 1)
        gmm = mixture.GaussianMixture(n_components = n_components, covariance_type = "full", random_state = None, n_init=100, max_iter=5000)
        gmm.fit(x)

        if not gmm.converged_:
            raise Exception("Estimator didn't converge!")

        norms = [gmm.means_[:,i] ** 2 for i in range(len(axis_list))]
        norms = sum(norms) ** 0.5
        
        sort_idx = np.argsort(norms)
        gmm.means_ = gmm.means_[sort_idx]
        gmm.weights_ = gmm.weights_[sort_idx]
        gmm.covariances_ = gmm.covariances_[sort_idx]
        gmm.bic_ = gmm.bic(x)
        gmm.aic_ = gmm.aic(x)

        eigenvalues, unit_eigenvectors = linalg.eigh(gmm.covariances_[0])

        # get the log probability of the sample for each gaussian state of the model
        gmm.ll_ = gmm.predict_proba(x)
        return(gmm)
    
    def generatePopulations(self, posterior_cutoff = 0.99):
        gated_exp = self.gated_exp.data
        # for each population, create a population object
        # for each population, classify the population based on location
        # for each population, get statistics based on covariance
        # mind the scales
        hasGroup = gated_exp[gated_exp['XD_gauss_gate_MaxPosterior'] >= posterior_cutoff]
        noGroup = gated_exp[gated_exp['XD_gauss_gate_MaxPosterior'] < posterior_cutoff]

        grouped = hasGroup.groupby('XD_gauss_gate')
        for groupname, data_subset in grouped:
            frequency = float(len(data_subset))/len(gated_exp)
            self.populations.append(Population(groupname, data_subset, self.optimal_gmm, self.axis_list, self.scales, day = self.day, frequency=frequency))
        # If noGroup has some, generate a population called: no group


    def gateXD(self, settings, gate_name = "XD_gauss_gate"):

        experiment = self.exp

        event_assignments = pd.Series([None] * len(experiment), dtype="object")
        event_posteriors = pd.Series([0.0] * len(experiment))

        scales = self.scales
        axis_list = self.axis_list
        gmm = self.optimal_gmm
        new_experiment = experiment.clone()

        x = new_experiment.data.loc[:, axis_list]
        for idx, scale in enumerate(scales):
            x[axis_list[idx]] = scale(x[axis_list[idx]])
            #x = x[~(np.isnan(x[axis_list[idx]]))]

        # Remove nans that exist on possible axis
        x_na = x.isnull().any(axis=1)
        
        # X is a N by C matrix where N is the number of datapoints in the dataset and C is the number of dimensions you are gating on
        x = x.values
        x_na = x_na.values

        """
        ========================
        Gating via argmax log probability
        ========================

        This assigns each datapoint with a class label.
        """
        # generate a dummy array with the size of x
        predicted = np.full(len(x), -1, "int")
        # predict gives you the index or group of the max log-likelihood.
        predicted[~x_na] = gmm.predict(x[~x_na])

        # fill the dummyx array with predicted label, 1D length of all non-NA variables
        predicted_str = pd.Series(["(none)"] * len(predicted))

        # for each index, add the corresponding gate name in place. That is the group membership name.
        for c in range(0, self.est_n_comp):
            predicted_str[predicted == c] = "{0}_{1}".format(gate_name, c+1)
        predicted_str[predicted == -1] = "{0}_None".format(gate_name)

        """
        ========================
        Gating via posterior probability cutoff
        ========================

        """
        probability = np.full((len(x), self.est_n_comp), 0.0, "float")
        probability[~x_na, :] = gmm.predict_proba(x[~x_na, :])


        # Posteriors are by default, 0.
        posteriors = pd.Series([0.0]* len(predicted))

        # Only the highest posterior value (highest out of the predicted components) is included into the dataframe, corresponds to the class label
        for c in range(0, self.est_n_comp):
            posteriors[predicted == c] = probability[predicted == c,c]
        
        #posteriors.index = group_idx
        event_posteriors = posteriors

        # Full posterior tables
        prob_df = pd.DataFrame(probability)
        prob_df.columns = ["{}_Posterior".format(x) for x in range(probability.shape[1])]

        #new_experiment.data = pd.concat([new_experiment.data, prob_df], ignore_index = True, axis = 1)
        #new_experiment.data.columns = experiment.data.columns + prob_df.columns

        new_experiment.metadata = self.exp.metadata.copy()
        if self.est_n_comp == 1:
            new_experiment.add_condition(gate_name, "bool", event_assignments == "{0}_1".format(gate_name))
        else:
            new_experiment.add_condition(gate_name, "category", predicted_str)

        col_name = "{0}_MaxPosterior".format(gate_name)
        new_experiment.add_condition(col_name,"float", event_posteriors)

        for n in range(self.est_n_comp):
            post_name = "{}_Posterior".format(n)
            new_experiment.add_condition(post_name, "float", prob_df[post_name])

        self.data = new_experiment.data.copy()
        self.data[gate_name] = predicted_str.copy()
        self.gated_exp = new_experiment
        return(new_experiment)

    def gateSurface(self, xchannel, ychannel, ncomp, sigma=4, gate_type = 'quad', anchor = 'br'):
        """ Gates based on the Xchannel and Ychannel. Creates a polygon gate or a quad gate based on the polygon gate.

        Gating rules: get the left most polygon gate and set the quadgate on top left corner
        ?
        Arg:

        Returns:
        """
        gate_name = '{0!s}_{1!s}_{2!s}'.format(xchannel, ychannel, "gauss")
        xchannel + '_' + ychannel + str(ncomp)
        self.surface_gates[gate_name] = flow.GaussianMixture2DOp(  name = gate_name,
                                        xchannel = xchannel,
                                        xscale = "logicle",
                                        ychannel = ychannel,
                                        yscale = "logicle",
                                        num_components = ncomp,
                                        sigma = sigma)
        exp_live_only = self.exp_live.query("Live == True")
        self.surface_gates[gate_name].estimate(exp_live_only)
        p_results = self.surface_gates[gate_name].default_view().plot(exp_live_only, get_coords = anchor, get_stats = True)
        pdb.set_trace()
        self.stats['anchor_vertices'] = util.convert_coords(p_results[0])
        self.stats['anchor_mean'] = p_results[1]
        self.stats['anchor_covar'] = p_results[2]
        self.stats['anchor_xl'] = p_results[3]
        self.stats['anchor_yl'] = p_results[4]


        if 'quad' in gate_type:
            # Convert anchored gaussian gate into poly gate
            poly_gate_coords = util.convert_coords(p_results[0])
            poly_gate = flow.PolygonOp(name = gate_name, xchannel = xchannel, ychannel = ychannel, vertices = poly_gate_coords)
            gate_name = '{0!s}_{1!s}_{2!s}'.format(xchannel, ychannel, gate_type)
            self.surface_gate_coords[gate_name] = poly_gate_coords
            self.surface_gates[gate_name] = poly_gate
            exp_live = self.exp_live.query("Live == True and Isotype == False ")
            #test_data = poly_gate.apply(exp_live)
            #x,y,z = self.calcGradient(exp_live, xchannel, ychannel)
            #exp_live.data['density'] = z

            #pv = poly_gate.default_view(huefacet=gate_name)
            anchor_corner = {'br':'tl', 'bl':'tr'}

            # Convert poly gate to quad gate

            self._convertPolyToQuad(gate_name, anchor_corner[anchor], xchannel, ychannel)
            #q_view = self.surface_gates[gate_name].default_view(huefacet=gate_name)
            #pdb.set_trace()
            #quad_gated_data = self.surface_gates[gate_name].apply(exp_live)
            #com_string1 = [str(x) for x in self.com]
            #com_string = '_'.join(com_string1)
            #fig_path = "static/{0!s}/{1!s}_{2!s}.png".format(self.EXP_NAME, com_string,gate_name)

            #q_view.plot(quad_gated_data, gate_name, gradient = True, bsave=fig_path)
        plt.close('all')

    def saveGradient(self, xchannel, ychannel):
        exp_live = self.exp_live.query("Live == True and Isotype == False ")
        x,y,z = self.calcGradient(exp_live, xchannel, ychannel)
        exp_live.data['density'] = z
        self.exp_live = exp_live.clone()

    def plotSurfaceQuad(self, xchannel, ychannel):
        exp_live = self.exp_live.query("Live == True and Isotype == False ")
        gate_name = '{0!s}_{1!s}_quad'.format(xchannel, ychannel)
        q_view = self.surface_gates[gate_name].default_view(huefacet=gate_name)
        com_string1 = [str(x) for x in self.com]
        com_string = '_'.join(com_string1)
        com_string = com_string.replace(" ", "_")
        com_string = com_string.replace("+", "pos")
        fig_path = "static/{0!s}/{1!s}_{2!s}.svg".format(self.EXP_NAME, com_string,gate_name)

        q_view.plot(exp_live, gate_name, gradient = True, bsave=fig_path)
        return(fig_path)


    def _convertPolyToQuad(self,gate_name, gate_type, xchannel, ychannel):
        x_minmax = (min(self.surface_gate_coords[gate_name], key = lambda t: t[0])[0], max(self.surface_gate_coords[gate_name], key = lambda t: t[0])[0])
        y_minmax = (min(self.surface_gate_coords[gate_name], key = lambda t: t[1])[1], max(self.surface_gate_coords[gate_name], key = lambda t: t[1])[1])
        corners = { 'tl': (x_minmax[0], y_minmax[1]),
                    'tr': (x_minmax[1], y_minmax[1]),
                    'bl': (x_minmax[0], y_minmax[0]),
                    'br': (x_minmax[1], y_minmax[0])}

        if 'auto' in gate_type:
            target_corner = corners['tr']
        else:
            target_corner = corners[gate_type]

        quad_g = flow.QuadOp(name = gate_name, xchannel = xchannel, ychannel = ychannel, xthreshold = target_corner[0], ythreshold = target_corner[1])
        self.surface_gates[gate_name] = quad_g
        self.stats[gate_name] = (target_corner[0], target_corner[1])

    def updateSurfaceGates(self, ref_surface_gates):
        self.surface_gates = ref_surface_gates.copy()
        
        for key, value in ref_surface_gates.items():
            self.stats[key] = (value.xthreshold, value.ythreshold)

    def applySurfaceGates(self):
        self.exp_live = self.exp_live.query("Live == True and Isotype == False ")
        for gate in self.surface_gates.values():
            self.exp_live = gate.apply(self.exp_live)

        CD34_CD41 = self.CD34_name + '_' + self.CD41_name
        CD41_CD42 = self.CD41_name + '_' + self.CD42_name
        
        ## get stats for surface gates
        gate_names = [CD34_CD41+'_quad', CD41_CD42+'_quad']
        for gate_name in gate_names:
            gate_col = self.exp_live.data[gate_name].value_counts()/len(self.exp_live[gate_name])
            gate_dict = dict(zip(gate_col.index, gate_col.values))
            for x in range(1,5):
                key = "{0}_{1!s}".format(gate_name, x)
                if key not in gate_dict:
                    gate_dict[key] = 0.0
            self.stats[gate_name+'_tl'] = "{0:.2g}".format(gate_dict["{0}_1".format(gate_name)])
            self.stats[gate_name+'_bl'] = "{0:.2g}".format(gate_dict["{0}_3".format(gate_name)])
            self.stats[gate_name+'_tr'] = "{0:.2g}".format(gate_dict["{0}_2".format(gate_name)])
            self.stats[gate_name+'_br'] = "{0:.2g}".format(gate_dict["{0}_4".format(gate_name)])

    def applySurfaceGatesDead(self):
        self.exp_dead = self.exp_dead.query("Dead == True and Isotype == False ")
        for gate in self.surface_gates.values():
            self.exp_dead = gate.apply(self.exp_dead)

        CD34_CD41 = self.CD34_name + '_' + self.CD41_name
        CD41_CD42 = self.CD41_name + '_' + self.CD42_name

        ## get stats for surface gates
        gate_names = [CD34_CD41+'_quad', CD41_CD42+'_quad']
        for gate_name in gate_names:
            gate_col = self.exp_dead.data[gate_name].value_counts()/len(self.exp_dead[gate_name])
            gate_dict = dict(zip(gate_col.index, gate_col.values))
            for x in range(1,5):
                key = "{0}_{1!s}".format(gate_name, x)
                if key not in gate_dict:
                    gate_dict[key] = 0.0
            self.stats[gate_name+'_tl_d'] = "{0:.2g}".format(gate_dict["{0}_1".format(gate_name)])
            self.stats[gate_name+'_bl_d'] = "{0:.2g}".format(gate_dict["{0}_3".format(gate_name)])
            self.stats[gate_name+'_tr_d'] = "{0:.2g}".format(gate_dict["{0}_2".format(gate_name)])
            self.stats[gate_name+'_br_d'] = "{0:.2g}".format(gate_dict["{0}_4".format(gate_name)])

    def renameQuads(self):
        CD34_CD41 = self.CD34_name + '_' + self.CD41_name
        CD41_CD42 = self.CD41_name + '_' + self.CD42_name
        rename_dict = { CD34_CD41+'_quad_tl': 'CD34n_CD41p',
                        CD34_CD41+'_quad_tr': 'CD34p_CD41p',
                        CD34_CD41+'_quad_bl': 'CD34n_CD41n',
                        CD34_CD41+'_quad_br': 'CD34p_CD41n',
                        CD41_CD42+'_quad_tl': 'CD41n_CD42p',
                        CD41_CD42+'_quad_tr': 'CD41p_CD42p',
                        CD41_CD42+'_quad_bl': 'CD41n_CD42n',
                        CD41_CD42+'_quad_br': 'CD41p_CD42n'}
        for key in rename_dict.keys():
            self.stats[rename_dict[key]] = self.stats.pop(key)

        self.stats['CD34p'] = float(self.stats['CD34p_CD41n']) + float(self.stats['CD34p_CD41p'])
        self.stats['CD41p'] = float(self.stats['CD34p_CD41p']) + float(self.stats['CD34n_CD41p'])
        self.stats['CD42p'] = float(self.stats['CD41p_CD42p']) + float(self.stats['CD41n_CD42p'])
        self.stats['PRE-EXPANSION DAY'] = self.com[2]

    def renameQuadsDead(self):
        CD34_CD41 = self.CD34_name + '_' + self.CD41_name
        CD41_CD42 = self.CD41_name + '_' + self.CD42_name
        rename_dict = { CD34_CD41+'_quad_tl_d': 'CD34n_CD41p_d',
                        CD34_CD41+'_quad_tr_d': 'CD34p_CD41p_d',
                        CD34_CD41+'_quad_bl_d': 'CD34n_CD41n_d',
                        CD34_CD41+'_quad_br_d': 'CD34p_CD41n_d',
                        CD41_CD42+'_quad_tl_d': 'CD41n_CD42p_d',
                        CD41_CD42+'_quad_tr_d': 'CD41p_CD42p_d',
                        CD41_CD42+'_quad_bl_d': 'CD41n_CD42n_d',
                        CD41_CD42+'_quad_br_d': 'CD41p_CD42n_d'}
        for key in rename_dict.keys():
            self.stats[rename_dict[key]] = self.stats.pop(key)

        self.stats['CD34p_d'] = float(self.stats['CD34p_CD41n_d']) + float(self.stats['CD34p_CD41p_d'])
        self.stats['CD41p_d'] = float(self.stats['CD34p_CD41p_d']) + float(self.stats['CD34n_CD41p_d'])
        self.stats['CD42p_d'] = float(self.stats['CD41p_CD42p_d']) + float(self.stats['CD41n_CD42p_d'])
        self.stats['PRE-EXPANSION DAY'] = self.com[2]

    def plot_3D(self, xaxis, yaxis, zaxis):
        data = self.gated_exp
        xscale = 'logicle'
        yscale = 'logicle'
        zscale = 'logicle'

        xscale = cytoutil.scale_factory('logicle', data, xaxis)
        yscale = cytoutil.scale_factory('logicle', data, yaxis)
        zscale = cytoutil.scale_factory('logicle', data, zaxis)
        fig3d = Single3DFigure(data, xaxis, yaxis, zaxis, xscale=xscale, yscale=yscale, zscale=zscale)
        fig3d.add_3D_graph(data, xaxis, yaxis, zaxis, xscale=xscale, yscale=yscale, zscale=zscale)
        for pop in self.populations:
            fig3d.add_population(pop, 'pop', 'red')
        fig3d.render()
        pdb.set_trace()

    def plot_2D(self):
        data = self.gated_exp
        xaxis = 'FSC-A'
        yaxis = 'DAPI-A'
        xscale = 'logicle'
        yscale = 'logicle'

        xscale = cytoutil.scale_factory('logicle', data, xaxis)
        yscale = cytoutil.scale_factory('logicle', data, yaxis)

        fig = Single2DFigure(data,xaxis, yaxis, xscale=xscale, yscale=yscale)
        fig.add_scatter(data, xaxis, yaxis, xscale=xscale, yscale=yscale, huefacet='XD_gauss_gate')
        fig.render()

        xaxis = 'PE-A'
        yaxis = 'FITC-A'
        xscale = 'logicle'
        yscale = 'logicle'
        xscale = cytoutil.scale_factory('logicle', data, xaxis)
        yscale = cytoutil.scale_factory('logicle', data, yaxis)

        fig = Single2DFigure(data,xaxis, yaxis, xscale=xscale, yscale=yscale)
        current_graph=fig.add_scatter(data, xaxis, yaxis, xscale=xscale, yscale=yscale, huefacet='XD_gauss_gate')
        test_pop = self.populations[0]
        current_graph.add_2d_ellipsoid(test_pop.mean, test_pop.axes, 'PE-A','FITC-A', self.axis_list)
        fig.render()
    
        

    def estimateXD(self, settings, max_ncomp):
        """
        Creates GMM models given the number of components.
        Estimates means, weights, covars of each group

        """
        experiment = self.exp
        current_data=experiment.data

        axis_list = settings['COLORS'].keys()
        scale_types = settings['COLORS'].values()

        scales = []
        # lists all scales
        for idx,ax in enumerate(axis_list):
            scales.append(cytoutil.scale_factory(scale_types[idx], experiment, ax))
        self.scales = scales
        self.axis_list = axis_list
        gmms = {}
        bics = []
        aics = []
        gmms = []
        import time
        ncomps = range(1,max_ncomp+1)
        for ncomp in ncomps:
            t0 = time.time()
            print("Currently making GMM #", ncomp)
            current_gmm = self._makeXDGMM(current_data, ncomp, axis_list, scales)
            bics.append(current_gmm.bic_)
            aics.append(current_gmm.aic_)
            gmms.append(current_gmm)
            t1 = time.time()
            print("{} components took {} seconds".format(ncomp, t1-t0))

        elbow_x, elbow_y = util.elbow_criteria(ncomps, bics)
        bic_g = {}
        bic_g['x'] = ncomps
        bic_g['y'] = bics
        bic_g['x_lim'] = (0,max_ncomp)
        bic_g['x_label'] = r"$K$"
        bic_g['y_label'] = "BIC"
        bic_g['title'] = "BIC Model Comparison"
        com_string1 = [str(x) for x in self.com]
        com_string = '_'.join(com_string1)
        bic_g['fig_path'] = "reports/{0!s}/bics/{1!s}_3Dbic.png".format(self.EXP_NAME, com_string)
        util.lineplot(bic_g)

        aic_g = bic_g.copy()
        aic_g['y'] = aics
        aic_g['y_label'] = "AIC"
        aic_g['title'] = "AIC Model Comparison"
        aic_g['fig_path'] = "reports/{0!s}/bics/{1!s}_3Daic.png".format(self.EXP_NAME, com_string)
        util.lineplot(aic_g)
        self.est_n_comp = elbow_x
        self.optimal_gmm = gmms[elbow_x-1]
        return(elbow_x)
    
    def plot_pop_overlay(self):
        #eigenvector
        #eigenvalue
        #3D or 2D
        #mean point
        pass

    def plot_convex_hull(self):
        pass

    def animate_convex_hull(self):
        pass

    def scanPopulations(self, xchannel, ychannel, ncomp = range(1,11)):
        """
        Arg:

        Returns:
            tuple, best number of components and bic
            list of tuples, component and corresponding bic

        """
        gates = []
        bic = []
        print(self.com,self.day)
        for comp in ncomp:
            gate_name = '{0!s}_{1!s}_{2!s}'.format(xchannel, ychannel, ncomp)
            xchannel + '_' + ychannel + str(ncomp)
            gates.append(flow.GaussianMixture2DOp(  name = gate_name,
                                            xchannel = xchannel,
                                            xscale = "logicle",
                                            ychannel = ychannel,
                                            yscale = "logicle",
                                            num_components = comp,
                                            sigma = 4))
        plt.close('all')
        live_only = self.exp_live.query("Live == True")
        for gate in gates:
            gate.estimate(live_only)
            #d_results = gate.default_view().plot(self.live_exp_only, get_coords = 1, get_stats=True)
            bic.append(gate._gmms[True].bic_)
        elbow_x, elbow_y = util.elbow_criteria(ncomp, bic)
        bic_g = {}
        bic_g['x'] = ncomp
        bic_g['y'] = bic
        bic_g['x_lim'] = (0,10)
        bic_g['x_label'] = r"$K$"
        bic_g['y_label'] = "BIC"
        bic_g['title'] = "BIC Model Comparison"
        com_string1 = [str(x) for x in self.com]
        com_string = '_'.join(com_string1)
        bic_g['fig_path'] = "static/{0!s}/{1!s}_bic.png".format(self.EXP_NAME, com_string)
        util.lineplot(bic_g)
        return(elbow_x)


    def gateLiveDead(self, size=True, augmented=True):
        self._gateLiveCells(size=size, augmented=augmented)
        self._gateDeadCells()
        self.mergeLiveDead()

    def mergeLiveDead(self):
        live_only = self.exp_live.query("Live == True")
        dead_only = self.exp_dead.query("Dead == True")

        self.live_exp_only = live_only

        live_only.add_condition("Dead", "bool", pd.Series([False for x in range(len(live_only.data))]))
        dead_only.add_condition("Live", "bool", pd.Series([False for x in range(len(dead_only.data))]))

        live_dead = live_only.clone()
        live_dead.merge_events(dead_only.data, dead_only.conditions)

        self.exp_live_dead = live_dead

    def calcGradient(self, data, xchannel, ychannel):
        x, y, z = util.get_channel_data(data, xchannel, ychannel)
        return(x,y,z)

    def plotLive(self):
        com_string1 = [str(x) for x in self.com]
        com_string = '_'.join(com_string1)
        fig_path = "static/{0!s}/{1!s}_livegate.png".format(self.EXP_NAME, com_string)
        pv = self.live_gate.default_view(huefacet="Live")
        pv.plot(self.exp_live, bsave=fig_path)

    def plotDead(self):
        com_string1 = [str(x) for x in self.com]
        com_string = '_'.join(com_string1)
        fig_path = "static/{0!s}/{1!s}_deadgate.png".format(self.EXP_NAME, com_string)
        dpv = self.live_gate.default_view(huefacet="Dead")
        dpv.plot(self.exp_dead, bsave=fig_path)
