'''
TODO: 
- Set different bounds for each column
- Check if data is in range, if not, make it nan
'''

from MyMetamodel import * 
import numpy as np
import scipy.stats as stats

def counts_to_data(counts):
    """
    Converts a vector of counts to data.
    """
    assert type(counts) is list or type(counts) is np.ndarray
    K = len(counts)
    N = int(sum(counts))
    X = []
    for k in range(K):
        i = 0
        while i < counts[k]:
            X.append([k])
            i += 1

        assert i == counts[k]

    assert len(X) == N

    np.random.shuffle(X)
    X = np.array(X, dtype=float)

    return X

def data_to_counts(array, K):
    """
    Converts a data numpy array with integers from 1 to K to counts.
    """
    assert type(array) is np.ndarray and type(K) is int
    counts = np.zeros(K)
    noRows = len(array)
    for i in np.arange(noRows):
        if not np.isnan(array[i]):
            value = array[i]
            counts[value] =+1

    return counts

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
        and type(value) is not np.float64:
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

class CategoricalMetamodel(MyMetamodel):
    """Dirichlet-Categorical metamodel for MyMetamodel.

    The metamodel is named ``dirichlet_categorical``.
    
    Class Methods:
        def __init__(self,data,typespec,hypers=[]): 
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

    def __init__(self,data,typespec,hypers=[]): 
        super(CategoricalMetamodel, self).__init__(data,typespec)

        noCols = self.data.shape[1]
        if check_hypers(hypers, noCols):
            self.hypers = hypers
        else:
            self.hypers = self.initialize_hypers()

    def initialize_hypers(self):
        '''Initialize hyperparameters for current prior model. All columns have same prior.
        Outputs a dict mappling hyperparameter names to a list of hyperameters.'''
        noCols = self.data.shape[1]
        alpha = [.5 for _ in np.arange(noCols)]
        K = [7 for _ in np.arange(noCols)]
        return dict(dirichlet_alpha=alpha, K=K)
        
    def name(self): return 'dirichlet_categorical'
  
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

        weights = self.get_predictive_params()['weights'][colno]
        value = np.array(weights).argmax()
        confidence = np.array(weights).max()
        return value, confidence      
        
    def simulate(self, colnos, constraints=[], numpredictions=1):
        """Simulate 'colnos' from a generator, subject to 'constraints'.
        
        Returns a list of rows with values for the specified columns.
        
        'colnos'         - list of column numbers.
        'constraints'    - list of ''(colno, value)'' pairs.
        'numpredictions' - number of results to return.
        """ 
        check_constraints(constraints, colnos)

        weights = self.get_predictive_params()['weights']

        simulation = np.array([])
        for icol in colnos:
            # multinomial draw
            counts = np.array(np.random.multinomial(numpredictions, weights[icol]), dtype=int)
            simulation = np.append(simulation,counts_to_data(counts))

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

        weights = self.get_predictive_params()['weights']

        # Sum of logpdf, due to independence of columns
        logpdf_out = 0
        for i in np.arange(len(targets)):
            colno = targets[0][i]
            value = targets[1][i]
            logpdf_out += np.log(weights[colno][value])

        return logpdf_out

    def get_predictive_params(self):
        '''Get parameters for predictive posterior distribution for each column.
        Returns a dict mapping parameter names to a list of parameters.
        Predictive posterior is a Categorical distribution.
        '''
        alphas = np.array(self.get_posterior_hypers()['dirichlet_alpha'])
        K = self.get_posterior_hypers()['K']

        new_weights = []
        noCols = self.data.shape[1]
        for colno in np.arange(noCols):
            new_weights.append((alphas[colno]/alphas[colno].sum()).tolist())
            
        return dict(weights=new_weights, K=K)

    def get_posterior_hypers(self):
        '''Get hyperparameters for posterior distribution for each column.
        Returns a dict mapping parameter names to a list of parameters.
        Posterior distribution is Dirichlet.
        '''
        alphas = self.hypers['dirichlet_alpha']
        K = self.hypers['K']

        new_alphas = []
        noRows = np.sum(~np.isnan(self.data),axis=0)
        noCols = self.data.shape[1]
        for colno in np.arange(noCols):
            counts = data_to_counts(self.data[colno], K[colno])
            new_alphas.append(alphas[colno] + counts)

        return dict(dirichlet_alpha=new_alphas, K=K)


'''
#Test basic functionalities


from CategoricalMetamodel import *

X = np.tile(1., [100,1])
for [i,j] in [[0,0],[23,0],[44,0]]:
    X[i][j] = float('nan')

test = CategoricalMetamodel(X,'standard')

print(X)

print(test.simulate([0],numpredictions=100))

test.infer()

print(test.data)
print(test.estimate(1,2))
'''