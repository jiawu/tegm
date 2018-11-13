import re
import pdb
from scipy import linalg
import numpy as np
from scipy.stats import norm, chi2
from utility import get_2d_mask
import pandas as pd
import numpy as np

class Population:

    def __init__(self, group_name, df, gmm, axis_list, scales, frequency=0, day=0):
        self.day = day
        self.group_name = group_name
        self.df = df
        self.gmm = gmm
        self.scales = scales
        self.axis_list = axis_list
        p = re.compile(r'XD_gauss_gate_(\d+)')
        match = p.search(group_name)
        self.group_number = int(match.group(1))
        g_idx = self.group_number-1

        self.principle_axis = []

        means = gmm.means_[g_idx]
        descaled_means = [self.scales[idx].inverse(x) for idx,x in enumerate(means)]
        self.mean = dict(zip(self.axis_list,descaled_means))
        self.scaled_means = dict(zip(self.axis_list,means))
        self.weight = gmm.weights_[g_idx]
        self.covariance = gmm.covariances_[g_idx]

        self.eigenvalues, self.unit_eigenvectors = linalg.eigh(self.covariance)

        self.axlengths = np.sqrt(self.eigenvalues*chi2.ppf(.99, 3))
        self.axes = np.dot(self.unit_eigenvectors, np.diag(self.axlengths))
        
        # an ellipsoid is the following:
        # (x - v)^T A (x - v) = 1
        #self.the   ta = self.angle_between(self.unit_eigenvector, )
        #self.phi = self.angle_between()
        #self.tau = self.angle_between()
        #x being cd34
        #y being cd41
        #z being cd42

        #eig1_x
        #eig1_y
        #eig1_z
        vec_x = (0,1,0,0,0,0)
        vec_y = (0,0,0,0,1,0)
        vec_z = (0,0,0,0,0,1)

        eig1_x = self.angle_between(self.unit_eigenvectors[:,-1], vec_x)
        eig1_y = self.angle_between(self.unit_eigenvectors[:,-1], vec_y)
        eig1_z = self.angle_between(self.unit_eigenvectors[:,-1], vec_z)
        
        eig2_x = self.angle_between(self.unit_eigenvectors[:,-2], vec_x)
        eig2_y = self.angle_between(self.unit_eigenvectors[:,-2], vec_y)
        eig2_z = self.angle_between(self.unit_eigenvectors[:,-2], vec_z)

        eig3_x = self.angle_between(self.unit_eigenvectors[:,-3], vec_x)
        eig3_y = self.angle_between(self.unit_eigenvectors[:,-3], vec_y)
        eig3_z = self.angle_between(self.unit_eigenvectors[:,-3], vec_z)

        linearity = (self.eigenvalues[-1] - self.eigenvalues[-2])/self.eigenvalues[-1]
        planarity = (self.eigenvalues[-2] - self.eigenvalues[-3])/self.eigenvalues[-1]
        omnivariance = np.power(np.product(self.eigenvalues),1./len(self.eigenvalues))
        anisotropy = (self.eigenvalues[-1]-self.eigenvalues[0])/self.eigenvalues[-1]
        eigentropy = -(np.sum(self.eigenvalues*np.log(self.eigenvalues)))

        self.cov_34_41 = get_2d_mask('PE-A','FITC-A',self.axis_list, self.covariance)
        self.cov_34_41_eig, self.cov_34_41_eigvec = linalg.eigh(self.cov_34_41)
        ecc_34_41 = self.cov_34_41_eig[1]/self.cov_34_41_eig[0]

        self.cov_41_42 = get_2d_mask('FITC-A','PE-A',self.axis_list, self.covariance)
        self.cov_41_42_eig, self.cov_41_42_eigvec = linalg.eigh(self.cov_41_42)
        ecc_41_42 = self.cov_41_42_eig[1]/self.cov_41_42_eig[0]
        # with respect to specific axes 34, 41, 42
        percent_membership = frequency

        mean_features = {k+'_mean': v for k, v in self.scaled_means.items()}
        pop_features = {'weight':self.weight,
            'eig1_x':eig1_x,
            'eig1_y':eig1_y,
            'eig1_z':eig1_z,
            'eig2_x':eig2_x,
            'eig2_y':eig2_y,
            'eig2_z':eig2_z,
            'eig3_x':eig3_x,
            'eig3_y':eig3_y,
            'eig3_z':eig3_z,
            'linearity': linearity,
            'planarity':planarity,
            'omnivariance':omnivariance,
            'anisotropy':anisotropy,
            'eigentropy':eigentropy,
            'ecc_34_41': ecc_34_41,
            'ecc_41_42': ecc_41_42,
            'percent_membership': percent_membership,
            'cluster_idx': g_idx
        }
        pop_features.update(mean_features)
        self.pop_features = pop_features
        
        #self.ecc_34_41 = 
        #self.ecc_41_42
        # eccentricity
        # linearity = (eig1 - eig2)/eig1
        # planarity = (eig2 - eig3)/eig1
        # sphericity = eig3/eig1
        # omnivariance = 3rootsquare(eig1,eig2,eig3)
        # anisotropy = (eig1 - eig3)/eig1
        # eigentropy = -(sum of eigenvalues*ln(eigenvalues))
        # change of curvature = eig3/(eig1+eig2+eig3)
        # minimum volume ellipsoid
        # minimum covariance determinant

        # attribute clustering
        # Fisher sphericity: fisher 2010
        # f_sphere = (sum of eig^(2*r)/p)/(sum of eig^r/p)^2
        # r=2

        #omnivariance: (product of all eigenvectors)^(1/n)
        #isotropy: eig_n/eig_1
        # anisotropy: (eig_1-eig_n)/eig_1
        # dimensionality: alpha_d is (eig_d - eig_d + 1)/(eig_1), d < n, or isotropy if d=n
        # dimensional label: d: argmax dimensionality
        # component entropy: - (sum of eig_d * log_n* eig_d)
        # dimensional entropy: - (sum of alpha_d log_n alpha_d)

        # linearity, planarity, sphericity represent neighborhood's participation in higher dimensions
        # how much it spreads into each dimension

    def unit_vector(self,vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self,v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        result = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return result