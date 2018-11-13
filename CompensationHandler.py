import pdb
import cytoflow as flow
import utility as util
import numpy as np

class CompensationHandler:
    """
    Given an experiment and paths of compensation beads,
    it (1) determines what the compensation matrix should be
    and (2) applies the operation.
    """
    def __init__(self, experiment, voltage, all_comp_paths, best_comp_paths):
        self.bead_dicts = []
        self.voltage = voltage
        self.experiment = experiment
        self.all_comp_paths = all_comp_paths
        self.current_comp_paths = best_comp_paths
        
        # Outputs from all methods
        self.channels = None
        self.a = None
        self.a_inv = None
        self.comp_op = None
        self.spillover = None
        self.all_bead_exp = None
        self.bead_gate = None

        self.import_beads()
        self.estimate_gates()
        #self.remove_edge_events()
        self.subset_beads()
        self.estimate_compensation()
        
        self.comp_all_bead_exp = self.comp_op.apply(self.all_bead_exp)
        
        self.comp_bead_dicts = []
        for bead_dict in self.bead_dicts:
            new_bead_dict = bead_dict.copy()
            new_bead_dict['exp'] = self.comp_op.apply(bead_dict['exp'])
            self.comp_bead_dicts.append(new_bead_dict)

        
        self.comp_exp = self.comp_op.apply(self.experiment)

    def estimate_compensation(self):
        bl_op = flow.BleedthroughLinearOp()

        # Parsing the best compensation paths
        # To be a dict with the key = colors, value = path
        paths = self.current_comp_paths['PATH'].tolist()
        colors = self.current_comp_paths['TUBE NAME'].tolist()
        # append -A to the end of all the colors
        colors = ['{}-A'.format(x) for x in colors]

        controls=dict(zip(colors,paths))

        bl_op.controls = controls
        gated_exp = self.bead_dicts[0]['exp']
        bl_op.estimate(gated_exp, subset = "Bead == True")
        
        channels = list(set([x for (x, _) in bl_op.spillover.keys()]))
        a = [  [bl_op.spillover[(y, x)] if x != y else 1.0 for x in channels] for y in channels]
        
        a_inv = np.linalg.pinv(a)

        self.comp_op = bl_op
        self.spillover = bl_op.spillover
        self.a = np.array(a)
        self.a_inv = a_inv
        self.channels = channels

        return(self.comp_op)

    def import_beads(self):
        # Get the bead gates for each file in all_comp_paths
        tube_list = []
        for idx, row in self.all_comp_paths.iterrows():
            fp = row['PATH']
            color = str(row['TUBE NAME'])
            date = row['$DATE'].strftime('%m/%d/%Y')

            tube = flow.Tube(file = fp, conditions = {"Color" : color, "Date": date })
            tube_list.append(tube)

        # Merge all the data into experiment and then estimate the mixture
        
        import_op = flow.ImportOp(conditions = {'Color' : 'category', 'Date': 'category'},tubes = tube_list)
        comp_ex = import_op.apply()
        self.all_bead_exp = comp_ex
        return(comp_ex)
    
    def subset_beads(self):
        all_beads = self.all_bead_exp

        my_results = util.subset_by_condition(all_beads, ['Date', 'Color'])
        new_results = []
        for exp_dict in my_results:
            exp = exp_dict['exp']
            gated_exp = self.bead_gate.apply(exp)
            bead_dict = {   "Voltage": self.voltage,
                            "Color": exp_dict['Color'],
                            "Date": exp_dict['Date'],
                            "exp": gated_exp,
                            "bead_events": len(gated_exp[gated_exp.data['Bead']])
                        }

            new_results.append(bead_dict)
        
        sorted_results = sorted(new_results, key=lambda k:(k['Date'], k['Color']))
        self.bead_dicts = sorted_results
        return(self.bead_dicts)

    def estimate_gates(self):
        """
        Generates bead gates based on the union of all experiments.

        """
        comp_ex = self.all_bead_exp
        # Generate the bead gating
        g = flow.GaussianMixture2DOp(name = "Debris_Filter",
                                xchannel = "FSC-A",
                                xscale = "logicle",
                                ychannel = "SSC-A",
                                yscale = "logicle",
                                num_components = 4,
                                sigma = 3)
        g.estimate(comp_ex)

        bead_coords = g.default_view().plot(comp_ex,get_coords = 3)
        bead_coords = [tuple(l) for l in bead_coords]
        self.bead_gate = flow.PolygonOp(name = "Bead", xchannel = "FSC-A", ychannel = "SSC-A", vertices=bead_coords)

        # Create polygon gate out of gaussian gate
        return(self.bead_gate)

    def remove_edge_events(self):
        """ updates allbeads """
        
        all_beads = self.all_bead_exp
        
        all_beads.data = all_beads.data[all_beads.data['PE-A'] < 20000]
        all_beads.data = all_beads.data[all_beads.data['FITC-A'] < 20000]
        all_beads.data = all_beads.data[all_beads.data['APC-A'] < 20000]

        self.all_bead_exp.data = all_beads.data
        return(self.all_bead_exp)
