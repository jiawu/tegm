"""
CompensationReport is a series of plots that show FSC and SSC gates, then the intensity plots for each proposed color, then the compensation correction.

This will be a series of bokeh plots where you can highlight individual areas and have them show up in the 1D plot, and so forth.

"""
from Report import Report, Section, subSection, Graph
import cytoflow as flow
import pdb

class CompensationReport(Report):
    
    def __init__(self, title=""):
        super(CompensationReport,self).__init__(title)
    
    def add_comp_graphs(self, ch, title):
        """Takes a compensation handler and generates 3 subsections
        Section1: size bead/graphs with the polygons
        Section2: histograms for each bead
        Section3: compensated
        """
        current_section = self.add_section(title=title)

        s_section = current_section.add_subsection(title="Bead Gates")
        h_section=current_section.add_subsection("Histograms")
        m_section=current_section.add_subsection("Matrix", 0.75)

        c_section1 = current_section.add_subsection(title="Compensated Beads: APC-FITC")
        c_section2 = current_section.add_subsection(title="Compensated Beads: PE-FITC")
        c_section3 = current_section.add_subsection(title="Compensated Beads: PE-APC")
        #### For the bead gates:
        merged_data = ch.all_bead_exp
        xscale = flow.utility.scale_factory('logicle', merged_data, "FSC-A")
        yscale = flow.utility.scale_factory('logicle', merged_data, "SSC-A")

        xscale_beads = flow.utility.scale_factory('logicle', merged_data, "FITC-A")

        
        for bead_dict in ch.bead_dicts:
            color = bead_dict['Color']
            date = bead_dict['Date']
            data = bead_dict['exp'].data
            formatted_title = "{} {}".format(color, date)
            current_graph =s_section.add_scatter(data, xaxis='FSC-A', 
                                                yaxis='SSC-A', xscale=xscale , 
                                                yscale= yscale, huefacet='Bead', 
                                                title=formatted_title)
            current_graph.add_gate(ch.bead_gate.vertices)
            bead_stat = "{} Beads".format(bead_dict['bead_events'])
            current_graph.add_br_stat(bead_stat)
            add_colors = ['APC-A', 'PE-A', 'FITC-A']
            beads_only = data[data['Bead']==True]
            h_section.add_histogram(beads_only, xaxis_list=add_colors, xscale=xscale_beads, title=formatted_title)

        m_section.add_matrix(ch.a_inv, ch.channels)

        cmerged_data =ch.comp_all_bead_exp
        fitc_scale = flow.utility.scale_factory('logicle', cmerged_data, "FITC-A")
        apc_scale = flow.utility.scale_factory('logicle', cmerged_data, "APC-A")
        pe_scale = flow.utility.scale_factory('logicle', cmerged_data, "PE-A")


        for cbead_dict in ch.comp_bead_dicts:
            color = cbead_dict['Color']
            date = cbead_dict['Date']
            data = cbead_dict['exp'].data
            data = data[data['Bead']==True]
            formatted_title = "{} {}".format(color.replace("-","x"), date)
            current_graph =c_section1.add_scatter(data, xaxis='FITC-A', 
                                                yaxis='APC-A', xscale=fitc_scale , 
                                                yscale= apc_scale, 
                                                title=formatted_title)
            current_graph =c_section2.add_scatter(data, xaxis='FITC-A', 
                                                yaxis='PE-A', xscale=fitc_scale , 
                                                yscale= pe_scale, 
                                                title=formatted_title)
            current_graph =c_section3.add_scatter(data, xaxis='APC-A', 
                                                yaxis='PE-A', xscale=apc_scale , 
                                                yscale= pe_scale, 
                                                title=formatted_title)
            

            #current_graph = h_section.add_multihistogram(data, xaxis=['APC-A', 'FITC-A', 'PE-A'], xscale=norm_scale)
        
        #### Add the histograms

    
