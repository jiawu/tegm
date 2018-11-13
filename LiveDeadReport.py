from Report import Report, Section, subSection, Graph
import cytoflow as flow
import pdb
from utility import get_merged_exp

class LiveDeadReport(Report):
    
    def __init__(self, title=""):
        super(LiveDeadReport,self).__init__(title)

    def add_ld_graphs(self, fc_list, title=""):
        """Takes a compensation handler and generates 3 subsections
        Section1: size bead/graphs with the polygons
        Section2: histograms for each bead
        Section3: compensated
        """
        current_section = self.add_section(title=title)
        
        # create scales from merged data
        all_data = []
        for fc in fc_list:
            merged_data = fc.get_live_bool_data()
            all_data.append(merged_data)
        all_exps = get_merged_exp(all_data)

        xaxis = "FSC-A"
        yaxis = "DAPI-A"
        xscale = flow.utility.scale_factory('logicle', all_exps, xaxis)
        yscale = flow.utility.scale_factory('logicle', all_exps, yaxis)

        for fc in fc_list:            
            formatted_title = "{} {} {}".format(fc.grouping[0], fc.grouping[1], fc.grouping[2])
            fc_section = current_section.add_subsection(title=formatted_title)
            
            for metaexp in fc.metaexps:
                graph_title = "Day {}".format(metaexp.day)
                data = metaexp.exp_live_bool
                current_graph = fc_section.add_scatter(data, xaxis = xaxis,
                                                        yaxis=yaxis, xscale=xscale,
                                                        yscale=yscale, huefacet = "Live",
                                                        title=graph_title)
                current_graph.add_gate(metaexp.live_gate.vertices)
                current_graph.add_gate(metaexp.dead_gate.vertices)
                live_percent = metaexp.stats['viability']

                ld_stat = "{0:.1%} live cells".format(live_percent)
                current_graph.add_br_stat(ld_stat)