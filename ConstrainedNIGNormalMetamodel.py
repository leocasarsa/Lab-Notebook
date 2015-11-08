'''
TODO: 
- Set different bounds for each column
- Check if data is in range, if not, make it nan
'''

from MyMetamodel import * 
import numpy as np
import scipy.stats as stats

def check_hypers(hypers, noCols):
    confirm_check = True
    
    if hypers == []:
        confirm_check = False;
        return confirm_check

    if type(hypers) is not dict:
        raise TypeError("hypers should be a dict")

    keys = ['mu', 'nu', 'r', 's']

 
    for key in keys:
        if key not in hypers.keys():
            print("KeyWarning: missing key in hypers: %s" % key)
            confirm_check = False

    for key, value in hypers.iteritems():
        if key not in keys:
            print("KeyWarning: invalid hypers key: %s" % key)
            confirm_check = False
        elif len(hypers[key]) != noCols:
            print("KeyWarning: wrong size of hyper key: %s" % key)
            confirm_check = False

        if type(value) is not float \
        and type(value) is not numpy.float64:
            print("TypeWarning: %s should be float" % key)
            confirm_check = False

        if key in ['nu', 'r', 's']:
            if value <= 0.0:
                print("ValueWarning: hypers[%s] should be greater than 0" % key)
                confirm_check = False

        if ~confirm_check:
            print('Reinitializing hyperparameters to standard values.')

        return confirm_check


def check_targets(targets, noCols):
    '''Check if targets is a list of pairs (colNo,value), 
    with at most one pair per column.'''
    # Check colNo < noCols
    # targets_colnos = np.array([col for (col,_) in targets])
    # Check repetition colNo
    # Check list of pairs
    print('Targets check was not implemented yet')

def check_constraints(constraints, noCols, targets=[]):
    '''Check if constraints is a list of pairs (colNo,value), 
    with at most one pair per column, and no overlap with targets.
    targets can be pairs or colnos?'''

    # Check colNo < noCols
    # Check repetition colNo
    # Check list of pairs
    print('Constraints check was not implemented yet')

def check_single_column(colno):
    '''Check if colno is one integer.'''
    print('Single Column check was not implemented yet')

class ConstrainedNIGNormalMetamodel(MyMetamodel):
    """Constrained Normal-Inverse-Gamma-Normal metamodel for MyMetamodel.

    The metamodel is named ``constrained_nig_normal``.
    
    Class Methods:
        __init__(self,data,typespec)
        name(self)
        infer(self, threshold, numsamples=None)
        predict(self, colno, threshold, numsamples=None)
        predict_confidence(self, colno, numsamples=None)
        simulate(self, colnos, constraints, numpredictions=1)
        estimate(self, colno0, colno1)
        column_mutual_information(self, colno0, colno1, numsamples=None)
        logpdf(self, targets, constraints)

    Reference: http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    """

    def __init__(self,data,typespec,hypers=[],bounds=[float('-inf'), float('inf')]): 
        super(ConstrainedNIGNormalMetamodel, self).__init__(data,typespec)
        self.bounds = bounds

        noCols = self.data.shape[1]
        if check_hypers(hypers, noCols):
            self.hypers = hypers
        else:
            self.hypers = self.initialize_hypers()

    def initialize_hypers(self):
        '''Initialize hyperparameters for current prior model. All columns have same prior.
        Outputs a dict mappling hyperparameter names to a list of hyperameters.'''
        
        # Prior mean should be inside range
        lower_bound, upper_bound = self.bounds
        if lower_bound == float('-inf'):
            if upper_bound == float('inf'):
                mean = 0.0
            else:
                mean = min(upper_bound,0)
        else:
            if upper_bound == float('inf'):
                mean = max(lower_bound,0)
            else:
                mean = (lower_bound + upper_bound) / 2

        noCols = self.data.shape[1]
        m = [mean for _ in np.arange(noCols)]
        V = a = b = [1.0 for _ in np.arange(noCols)]
        return dict(loc=m, V=V, shape=a, scale=b)
        
    def name(self): return 'constrained_nig_normal'
  
    def infer(self, threshold=0, numsamples=None):
        """Predict a value for each missing value in self.data."""
        nanrows, nancols = np.where(np.isnan(self.data))
        for i in np.arange(nancols.size):
            prediction = self.predict(nancols[i], threshold, numsamples)
            self.data[nanrows[i],nancols[i]] = prediction
    
    def predict(self, colno, threshold, numsamples=None):
        """Predict a value for a column, if confidence is high enough."""
        value, confidence = self.predict_confidence(colno, numsamples=numsamples)
        if confidence <= threshold:
            return float('nan')
        return value
    
    def predict_confidence(self, colno, numsamples=None):
        """Predict a value for a column and return confidence"""
        # What is confidence exactly?
        check_single_column(colno)

        predictive_mean = self.get_predictive_params()['loc'][colno]
        lower_bound, upper_bound = self.bounds
        
        # Predicted value inside range
        value = lower_bound
        if lower_bound <= predictive_mean:
            value = upper_bound
            if predictive_mean <= upper_bound:
                value = predictive_mean

        confidence = 1 #closed-form posterior
        return value, confidence
        
        
    def simulate(self, colnos, constraints=[], numpredictions=1):
        """Simulate 'colnos' from a generator, subject to 'constraints'.
        
        Returns a list of rows with values for the specified columns.
        
        'colnos'         - list of column numbers.
        'constraints'    - list of ''(colno, value)'' pairs.
        'numpredictions' - number of results to return.
        
        Rejection sampling for simulated values staying within bounds.
        """ 
        check_constraints(constraints, colnos)

        predictive_params = self.get_predictive_params()
        scale = predictive_params['scale']
        df = predictive_params['degrees_of_freedom']
        loc = predictive_params['loc']

        lower_bound, upper_bound = self.bounds

        simulation = np.array([])
        for icol in colnos:
            sample_col = np.array([])
            for ipred in np.arange(numpredictions):
                in_range = False
                while not(in_range): # Reject samples out of range 
                    sample = stats.t.rvs(df[icol], loc=loc[icol], scale=scale[icol])
                    if lower_bound <= sample and sample <= upper_bound:
                        in_range = True
                sample_col = np.append(sample_col, sample) 
            simulation = np.append(simulation, sample_col)

        return colnos, simulation
    
    def estimate(self, colno0, colno1):
        """Compute Dependence Probability of <col0> with <col1>"""
        # check_col(colno0, colno1)
        dep_prob = 0
        return dep_prob
    
    def column_mutual_information(self, colno0, colno1, numsamples=None):
        """Compute ``MUTUAL INFORMATION OF <col0> WITH <col1>``."""
        mutual_info = 0   
        return mutual_info
    
    def logpdf(self, targets, constraints):
        """Compute (predictive?) logpdf.
        
                'constraints'    - list of ''(colno, value)'' pairs.
                'targets'        - list of ''(colno, value)'' pairs.
        """
        check_targets(targets, noCols)
        check_constraints(constraints, targets, noCols)

        predictive_params = self.get_predictive_params()
        scale = predictive_params['scale']
        df = predictive_params['degrees_of_freedom']
        loc = predictive_params['loc']

        lower_bound, upper_bound = self.bounds

        # logpdf is the same as the student-t, except for a normalization factor.
        logpdf_out = 0
        for i in np.arange(len(targets)):
            colno = targets[0][i]
            value = targets[1][i]
            normalization = (stats.t.cdf(upper_bound,  df[colno], loc=loc[colno], scale=scale[colno]) 
                - stats.t.cdf(lower_bound,  df[colno], loc=loc[colno], scale=scale[colno]))
            logpdf_out += np.log(stats.t.pdf(value, df[colno], loc=loc[colno], scale=scale[colno]) / normalization)

        return logpdf_out

    def get_predictive_params(self):
        '''Get parameters for predictive posterior distribution for each column.
        Returns a dict mapping parameter names to a list of parameters.
        '''
        posterior_hypers = self.get_posterior_hypers()
        m = posterior_hypers['loc']
        V = posterior_hypers['V']
        shape = posterior_hypers['shape']
        scale = posterior_hypers['scale']

        t_scale, df, t_loc = [], [], []
        noCols = self.data.shape[1]
        for i in np.arange(noCols):
            t_loc.append(m[i])
            df.append(2. * shape[i])
            t_scale.append(scale[i] * (1 + V[i]) / shape[i])

        return dict(scale=t_scale,degrees_of_freedom=df,loc=t_loc)

    def get_posterior_hypers(self):
        '''Get hyperparameters for posterior distribution for each column.
        Returns a dict mapping parameter names to a list of parameters.
        '''
        m = self.hypers['loc']
        V = self.hypers['V']
        shape = self.hypers['shape']
        scale = self.hypers['scale']

        avg_data = np.nanmean(self.data, axis=0)
        squared_sum = np.nansum(self.data**2, axis=0)

        new_m, new_V, new_shape, new_scale = [],[],[],[]
        noRows = np.sum(~np.isnan(self.data),axis=0)
        noCols = self.data.shape[1]
        for i in np.arange(noCols):
            new_V.append(1 / (noRows[i] + 1/V[i]))
            new_m.append(new_V[i] * (noRows[i]*avg_data[i] + m[i]/V[i]))
            new_shape.append(shape[i] + noRows[i]/2)
            new_scale.append(scale[i] + .5 * (m[i]**2 /V[i] + squared_sum[i] - new_m[i]**2 /new_V[i]))

        return dict(loc=new_m, V=new_V, shape=new_shape, scale=new_scale)


'''
#Test basic functionalities

# from ConstrainedNIGNormalMetamodel import *

X = np.tile(9., [10,5])
for [i,j] in [[0,0],[0,2],[1,3],[3,4],[4,4],[5,1]]:
    X[i][j] = float('nan')

test = ConstrainedNIGNormalMetamodel(X,'standard', bounds=[0,8])

print(test.simulate([1,2,3],numpredictions=100))

test.infer()
print(test.data)

print(test.estimate(1,2))
'''