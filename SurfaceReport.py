from Report import Report, Section, subSection, Graph
import cytoflow as flow
import pdb
from utility import get_merged_exp

class SurfaceReport(Report):
    
    def __init__(self, title=""):
        super(SurfaceReport,self).__init__(title)

    def add_surface_graphs(self, fc_list, xaxis_marker, yaxis_marker, title="", subset_list=None):
        """Takes a compensation handler and generates 3 subsections
        """
        current_section = self.add_section(title=title)
        
        # create scales from merged data
        all_data = []
        for fc in fc_list:
            fc.check_channels()
            merged_data = fc.get_live_data()
            # make sure the channels are consistent for a certain marker for each metaexp
            all_data.append(merged_data)

        all_exps = get_merged_exp(all_data)

        # Convert x axis marker to y axis marker
        # make sure the channels are consistent for a certain marker

        # I'm not adding this functionality in where the channels are different right now. F that.
        # Print a message when channels are different

        xaxis = fc.channel_dict[xaxis_marker]
        yaxis = fc.channel_dict[yaxis_marker]

        xscale = flow.utility.scale_factory('logicle', all_exps, xaxis)
        yscale = flow.utility.scale_factory('logicle', all_exps, yaxis)

        for fc in fc_list:         
            formatted_title = "{} {} {}".format(fc.grouping[0], fc.grouping[1], fc.grouping[2])
            fc_section = current_section.add_subsection(title=formatted_title)
            for metaexp in fc.metaexps:
                
                graph_title = "Day {}".format(metaexp.day)
                data = metaexp.exp_live_dead
                
                if subset_list:
                    for subset,subset_value in subset_list:
                        data = data[data[subset] == subset_value]

                current_graph = fc_section.add_scatter(data, xaxis = xaxis,
                                                        yaxis=yaxis, xscale=xscale,
                                                        yscale=yscale, huefacet = "",
                                                        title=graph_title)
                
                # MetaExp has an attr in self.stats[gate_name]
                # where gate_name is PE-A_FITC_A_quad for example
                # this returns a tuple value where the 0 is the x threshold
                # and 1 is the y threshold for a quad gate.

                gate_name = "{}_{}_quad".format(xaxis, yaxis)
                x_threshold, y_threshold = metaexp.stats[gate_name]
                current_graph.add_quad_gate(x_threshold, y_threshold)

                #live_percent = metaexp.stats['viability']

                #ld_stat = "{0:.1%} live cells".format(live_percent)

                print(formatted_title, len(data))
                br_stat = (data[(data[xaxis] >= x_threshold) & (data[yaxis] < y_threshold)].count()/data.count())[xaxis]
                tr_stat = (data[(data[xaxis] >= x_threshold) & (data[yaxis] >= y_threshold)].count()/data.count())[xaxis]
                bl_stat = (data[(data[xaxis] < x_threshold) & (data[yaxis] < y_threshold)].count()/data.count())[xaxis]
                tl_stat = (data[(data[xaxis] < x_threshold) & (data[yaxis] >= y_threshold)].count()/data.count())[xaxis]
                current_graph.add_br_stat("{0:.1%}".format(br_stat))
                current_graph.add_tr_stat("{0:.1%}".format(tr_stat))
                current_graph.add_bl_stat("{0:.1%}".format(bl_stat))
                current_graph.add_tl_stat("{0:.1%}".format(tl_stat))