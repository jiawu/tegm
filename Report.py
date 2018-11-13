import math
import pdb

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import norm, chi2
from utility import get_2d_mask

import matplotlib.path as path
import matplotlib.patches as patches
import matplotlib.transforms as transforms

rcParams['text.usetex'] = False

class Report(object):
    def __init__(self, title=""):
        self.sections = []
        self.title = title
    
    def add_section(self, title=""):
        self.current_section = Section(title)
        self.sections.append(self.current_section)
        return(self.current_section)
    
    def render(self, savepath, same_size = False, shared_axis = True, contour=True, maxcols=4):
        
        # Get number of sections
        # number of subsections
        # number of graphs in each subsection

        n_sections = len(self.sections)
        n_subsections = [len(x.subsections) for x in self.sections]
        n_graphs = [len(y.graphs) for x in self.sections for y in x.subsections]
        
        """
        if any([x > maxcols for x in n_graphs]):
            # ie if there are many graphs in each section
            subsec_nrows = [round(x/float(maxcols)) for x in n_graphs]
            total_rows = int(sum(subsec_nrows))
            total_cols = maxcols
        else:
            # ie if there are 3 graphs in each section
            total_rows = sum(n_subsections)
            total_cols = max(n_graphs)     
        """       
        plt.close("all")
        fig = plt.figure(figsize=(10,18))
        fig.text(0.5,0.98,self.title.replace('_',' '), fontsize=18, ha="center")

        gs = gridspec.GridSpec(n_sections, 1)
        running_hratio = 1
        for i,section in enumerate(self.sections):

            # invisible subplots to get the axis alignment right
            #sec_ax.text(0.5, 1.1, section.title,horizontalalignment="center", transform=sec_ax.transAxes, fontsize=15)

            top_point = (1.0/n_sections)*0.96*(float(i)*2+1)
            fig.text(0.5,top_point,section.title.replace('_',' '), fontsize=15, ha="center")

            n_sub= len(section.subsections)
            height_ratios = [x.h_ratio for x in section.subsections]
            gs_s = gridspec.GridSpecFromSubplotSpec(n_sub,1, subplot_spec=gs[i], height_ratios=height_ratios, hspace=0.2)
            
            section_height = 1.0/sum(height_ratios)
            current_mid = 0
            for j,subsection in enumerate(section.subsections):
                
                if same_size:
                    total_rows = int(math.ceil(max(n_graphs)/float(maxcols)))
                    total_cols = maxcols

                else:
                    if len(subsection.graphs) > maxcols:
                        total_rows = int(math.ceil(len(subsection.graphs)/float(maxcols)))
                        total_cols = maxcols
                    else:
                        total_rows = 1
                        total_cols = len(subsection.graphs)

                gs_sg = gridspec.GridSpecFromSubplotSpec(total_rows, total_cols, subplot_spec=gs_s[j])
                for k,graph in enumerate(subsection.graphs):
                    col_idx = int(k%total_cols)
                    row_idx = int(k/total_cols)
                    #print(row_idx,col_idx)
                    ax = plt.subplot(gs_sg[row_idx,col_idx])
                    graph.render(ax)

                #subsec_ax = plt.subplot(gs_s[j])
                #midpoint = (1.0/n_sub)*0.5*(float(j)*2+1)
                #subsecax=plt.gca()
                current_mid = current_mid + 0.5*(section_height*height_ratios[j])
                fig.text(0.01,1-current_mid,subsection.title,rotation=90, fontsize=15, va='center')
                #ax.subplot
                current_mid = current_mid + 0.5*(section_height*height_ratios[j])

                                
        plt.tight_layout(pad=7,w_pad=5, h_pad=1)
        plt.subplots_adjust(wspace=0.25, hspace=1)      
        fig.savefig(savepath)
        plt.close(fig)

class Section(object):
    def __init__(self, title=""):
        self.subsections = []
        self.title = title
    
    def add_subsection(self, title="",h_ratio=1 ):
        self.current_subsection = subSection(title, h_ratio)
        self.subsections.append(self.current_subsection)
        return(self.current_subsection)
        

class subSection(object):
    def __init__(self, title="",h_ratio= 1):
        self.graphs = []
        self.title = title
        self.h_ratio=h_ratio
    
    def add_scatter(self, data, xaxis, yaxis, xscale=None, yscale=None, huefacet = None, title = ""):
        self.current_graph = ScatterGraph(data, xaxis, yaxis, xscale, yscale, huefacet, title)
        self.graphs.append(self.current_graph)
        return(self.current_graph)

    def add_histogram(self, data, xaxis_list = [], xscale=None, title=""):
        self.current_graph = HistoGraph(data, xaxis_list, xscale, title)
        self.graphs.append(self.current_graph)
        return(self.current_graph)

    def add_matrix(self, mat, channels, title=""):
        self.current_graph = MatrixGraph(mat, channels, title)
        self.graphs.append(self.current_graph)
        return(self.current_graph)

class Single3DFigure(object):
    def __init__(self, data, xaxis, yaxis, zaxis, xscale=None, yscale=None, zscale=None, title=""):
        self.title = title
        self.graphs = []
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.zaxis = zaxis
        self.xscale = xscale
        self.yscale = yscale
        self.zscale = zscale

        self.data = data
        self.cmap = dict()
        self.populations = []

    def render(self):
        plt.close("all")
        fig = plt.figure(figsize=(10,18))
        ax=Axes3D(fig)
        #ax = fig.add_subplot(111, projection='3D')
        #self.graphs[0].render(ax)
        current_graph = Graph3D(self.data,self.xaxis,self.yaxis,self.zaxis, self.xscale, self.yscale, self.zscale, self.cmap, self.populations)
        current_graph.render(ax)

    def add_3D_graph(self,data, xaxis, yaxis, zaxis, xscale=None, yscale=None, zscale=None, title=""):
        self.current_graph = Graph3D(data, xaxis, yaxis, zaxis, xscale=xscale, yscale=yscale, zscale=zscale, title="")
        self.graphs.append(self.current_graph)

    def add_population(self, population, pop_name, color):
        self.current_graph.populations.append(population)
        #self.cmap[pop_name] = color

class Single2DFigure(object):
    def __init__(self, data, xaxis, yaxis, xscale=None, yscale=None, title=""):
        self.title = title
        self.graphs = []
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.xscale = xscale
        self.yscale = yscale

        self.data = data
        self.cmap = dict()
        self.populations = []

    def render(self):
        plt.close("all")
        fig = plt.figure(figsize=(10,18))
        ax=fig.add_subplot(111)
        #ax = fig.add_subplot(111, projection='3D')
        #self.graphs[0].render(ax)
        current_graph = self.graphs[0]
        current_graph.render(ax)

    def add_scatter(self, data, xaxis, yaxis, xscale=None, yscale=None, huefacet = None, title = ""):
        self.current_graph = ScatterGraph(data, xaxis, yaxis, xscale, yscale, huefacet, title)
        self.graphs.append(self.current_graph)
        return(self.current_graph)

class Graph(object):
    def __init__(self,data,gtype, title):
        self.data=data
        self.type = gtype
        self.title = title

class Graph3D(Graph):
    def __init__(self, data, xaxis, yaxis, zaxis, xscale=None, yscale=None, zscale=None, colors=None, title="", populations=[]):
        super(Graph3D, self).__init__(data, gtype="3d", title=title)
        self.type = "3d"
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.zaxis = zaxis

        self.xscale = xscale
        self.yscale = yscale
        self.zscale = zscale
        self.title = title
        self.cmap = colors
        self.populations = populations

    def render(self, ax):
        df = self.data
        xscale = self.xscale
        yscale = self.yscale
        zscale = self.zscale

        xaxis = self.xaxis
        yaxis = self.yaxis
        zaxis = self.zaxis

        scale_data_tup = zip( [xaxis, yaxis, zaxis],[xscale,yscale,zscale])
        scaled_data = []
        for aaxis, ascale in scale_data_tup:
            datum = self.data[aaxis].values
            scaled_datum = ascale(datum)
            scaled_data.append(scaled_datum)

        #for sub_pop_name, color in self.cmap.keys():
        #    sub_pop = self.data.query('sub_pop_name == @sub_pop_name')

        #ax.scatter(scaled_data[0], scaled_data[1], scaled_data[2], s=1, alpha=0.1)
        ax.set_axis_bgcolor('white')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.zaxis.label.set_color('black')
        ax.tick_params(axis='x',colors='black')
        ax.tick_params(axis='y',colors='black')
        ax.tick_params(axis='z',colors='black')
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        ax.set_xscale('logicle', **xscale.mpl_params)
        ax.set_yscale('logicle', **yscale.mpl_params)
        ax.set_zscale('logicle', **zscale.mpl_params)
        #ax.set_xlim(0,1)
        #ax.set_ylim(0,1)

        # plot the means of each population as a red dot
        colors = ['red', 'blue', 'green']
        for idx,pop in enumerate(self.populations):
            mean_dict = dict(pop.mean)
            mean_x = mean_dict[xaxis]
            mean_y = mean_dict[yaxis]
            mean_z = mean_dict[zaxis]

            #ax.scatter(mean_x,mean_y,mean_z, s=12, color='red', alpha=1)
            #print(colors[idx])
            ax.scatter(pop.df[xaxis],pop.df[yaxis], pop.df[zaxis], s=4, color=colors[idx], alpha=0.1)
            parsed_center = [mean_x,mean_y,mean_z]
            x_idx=pop.axis_list.index(xaxis)
            y_idx=pop.axis_list.index(yaxis)
            z_idx=pop.axis_list.index(zaxis)
            row_idx = np.array([x_idx, y_idx, z_idx])
            
            parsed_axes = pop.axes[row_idx[:,None], row_idx]

        print('done!')
        
            #self.__plot_ellipsoid_3D(ax, parsed_center, parsed_axes)
    
    def __plot_ellipsoid_3D(self, axes, center, radii, rotation, plotAxes = True):
        radii = np.column_stack([self.xscale.inverse(axes[:,0]),self.yscale.inverse(axes[:,1]), self.zscale.inverse(axes[:,2])])
        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)

        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
        

        for i in range(len(x)):
            for j in range(len(x)): 
                x[i,j], y[i,j], z[i,j] = center + np.dot([x[i,j],y[i,j],z[i,j]],rotation)
        
        if plotAxes:
            # make some purdy axes
            axes = np.array([[radii[0],0.0,0.0],
                             [0.0,radii[1],0.0],
                             [0.0,0.0,radii[2]]])
            # rotate accordingly
            for i in range(len(axes)):
                axes[i] = np.dot(axes[i], rotation)


            # plot axes
            for p in axes:
                X3 = np.linspace(-p[0], p[0], 100) + center[0]
                Y3 = np.linspace(-p[1], p[1], 100) + center[1]
                Z3 = np.linspace(-p[2], p[2], 100) + center[2]
                axes.plot(X3, Y3, Z3, color='red')
        
        pdb.set_trace()
        axes.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='#2980b9', alpha=0.5)

class HistoGraph(Graph):
    def __init__(self, data, xaxis_list, xscale=None, title=""):
        super(HistoGraph, self).__init__(data, gtype="histo", title=title)
        self.type = "histo"
        self.xaxis_list = xaxis_list
        self.xscale = xscale
        self.title = title
    
    def render(self, ax):
        df = self.data
        xscale = self.xscale
        
        x_min = 0
        x_max = 0
        h_list = []
        for xax in self.xaxis_list:
            x = self.data[xax].values
            scaled_data = xscale(x)
            new_x_min = min(scaled_data)
            new_x_max = max(scaled_data)
            if new_x_min < x_min:
                x_min = new_x_min
            
            if new_x_max > x_max:
                x_max = new_x_max
            h_list.append(self._scottsrule(scaled_data))
        
        num_bins = int(round(np.mean(h_list)))
        num_bins = max(min(num_bins, 1000), 50)
        bin_width = (x_max - x_min) / num_bins
        bins = xscale.inverse(np.arange(x_min, x_max, bin_width))
        bins = np.append(bins, xscale.inverse(x_max))

        for xax in self.xaxis_list:
            x = self.data[xax].values  
            scaled_data = xscale(x)
            ax.hist(x, bins=bins, alpha=0.3, color = self._cmapping(xax))
            ax.set_xscale('logicle', **xscale.mpl_params)
            ax.set_yscale('log')

            ax.tick_params(which='both', top=False, right=False, direction='out', labelsize=5)
        
        ax.set_title(self.title, fontsize=5)
        
    def _cmapping(self,channel):
        colormap = {"APC-A": 'red',
                    "PE-A": 'yellow',
                    "FITC-A": 'green'} 
        return(colormap[channel])  
    def _scottsrule(self, data):
        std = np.std(data, ddof=len(data)-1)
        h = (3.5*std)/(len(data)^(1/3))
        return(h)

class MatrixGraph(Graph):
    def __init__(self, mat, channels, title =""):
        super(MatrixGraph,self).__init__(mat,gtype="mat",title=title)
        self.type = "mat"
        self.title = title
        self.channels = channels
    
    def render(self, ax):
        cax=ax.imshow(self.data, interpolation="none", cmap='spring')
        ax.set_aspect('equal')
        fig=plt.gcf()
        fig.colorbar(cax, orientation="vertical")

        for (j,i),label in np.ndenumerate(self.data):
            new_label = "{0:.2E}".format(label)
            ax.text(i,j,new_label,ha='center',va='center')

        ax.set_xticks(np.arange(self.data.shape[1]), minor=False)
        ax.set_yticks(np.arange(self.data.shape[0]), minor=False)
        #ax.invert_yaxis()

        #lebels
        ax.set_xticklabels(self.channels, minor=False)
        ax.set_yticklabels(self.channels, minor=False)

class ScatterGraph(Graph):
    def __init__(self, data, xaxis, yaxis, xscale=None, yscale=None, huefacet = None, title =""):
        super(ScatterGraph,self).__init__(data,gtype="scatter",title=title)
        self.type = "scatter"
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.xscale = xscale
        self.yscale = yscale
        self.gates = []
        self.quad_gates = []
        self.huefacet = huefacet
        self.title = title
        self.br_stats = []
        self.tr_stats = []
        self.bl_stats = []
        self.tl_stats = []

    def add_2d_ellipsoid(self,center, axes, xaxis,yaxis, axis_list):
        n=100
        
        u = np.linspace(0.0, 2.0 * np.pi)
        v = np.linspace(0.0, np.pi)
        z = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        x = np.outer(np.ones_like(u), np.cos(v))

        center_2d = [center[xaxis],center[yaxis]]
        
        #get_2d_mask(xaxis, yaxis, axis_list, axes)
        x_idx = axis_list.index(xaxis)
        y_idx = axis_list.index(yaxis)

        mask = np.array([x_idx,y_idx])
        axes_2d = axes[mask[:,None], mask]
        for i in range(len(x)):
            for j in range(len(x)):
                x[i,j], y[i,j] = center_2d + np.dot(axes_2d,[x[i,j],y[i,j]])
        verts = zip(x,y)

        self.gates.append(verts)


    def _plot_ellipse(self, center, width, height, angle, **kwargs):
        tf = transforms.Affine2D() \
            .scale(width * 0.5, height * 0.5) \
            .rotate_deg(angle) \
            .translate(*center)

        tf_path = tf.transform_path(path.Path.unit_circle())
        v = tf_path.vertices
        v = np.vstack((self.xscale.inverse(v[:, 0]), self.yscale.inverse(v[:, 1]))).T

        scaled_path = path.Path(v, tf_path.codes)
        scaled_patch = patches.PathPatch(scaled_path, **kwargs)
        plt.gca().add_patch(scaled_patch)
        return(v)

    def add_gate(self, vertices):
        self.gates.append(vertices)
    
    def add_quad_gate(self, x_threshold, y_threshold):
        self.quad_gates.append((x_threshold, y_threshold))

    def add_br_stat(self, stat):
        self.br_stats.append(stat)

    def add_tr_stat(self, stat):
        self.tr_stats.append(stat)

    def add_bl_stat(self, stat):
        self.bl_stats.append(stat)

    def add_tl_stat(self, stat):
        self.tl_stats.append(stat)

    def render(self, ax):
        df = self.data
        x = self.data[self.xaxis].values
        y = self.data[self.yaxis].values

        xscale = self.xscale
        yscale = self.yscale
        
        if self.xscale:
            x = self.xscale(df[self.xaxis].values)
        if self.yscale:
            y = self.yscale(df[self.yaxis].values)
        
        if self.huefacet:
            keys = sorted(df[self.huefacet].unique())
            ncolors = len(keys)
            colors = []
            for co in range(ncolors):
                values = pylab.get_cmap('Paired')(co/float(ncolors))
                colors.append(values)
            colors = dict(zip(keys, colors))

            ax.scatter(x, y, c=df[self.huefacet].apply(lambda l: colors[l]), s=2, alpha=0.2)
        else:
            ax.scatter(x, y, s = 2, alpha=0.2)

        if len(self.gates) > 0:
            for vertices in self.gates:
                patch_vert = np.concatenate((np.array(vertices), np.array((0,0), ndmin = 2)))
                gate_poly = matplotlib.patches.PathPatch(matplotlib.path.Path(patch_vert, closed = True), edgecolor  = "black", linewidth = 2, fill = False)
                ax.add_patch(gate_poly)

        if len(self.quad_gates) > 0:
            for x_thresh, y_thresh in self.quad_gates:
                ax.axhline(y=y_thresh, linewidth=2)
                ax.axvline(x=x_thresh, linewidth=2)
        padding = 0.03
        self.__addstat(ax, self.br_stats, x=1-padding, y=0+padding, ha='right', va='bottom')
        self.__addstat(ax, self.tr_stats, x=1-padding, y=1-padding, ha='right', va='top')
        self.__addstat(ax, self.bl_stats, x=0+padding, y=0+padding, ha='left', va='bottom')
        self.__addstat(ax, self.tl_stats, x=0+padding, y=1-padding, ha='left', va='top')
        
        #ax.set_xscale('logicle', **xscale.mpl_params)
        #ax.set_yscale('logicle', **yscale.mpl_params)
        ax.tick_params(which='both', top=False, right=False, direction='out', labelsize=5)
        ax.set_title(self.title, fontsize=5)

    def __addstat(self, ax, stat_list, x, y, ha, va):
        if len(stat_list) > 0:
            full_text = ""
            for line in stat_list:
                full_text = full_text + line + '\n'
            full_text = full_text.rstrip('\n')
            ax.text(x,y,full_text,horizontalalignment=ha, verticalalignment=va, transform=ax.transAxes )
