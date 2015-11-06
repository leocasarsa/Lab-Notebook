import numpy as np
class NIGNormalMetamodel(MyMetamodel):
    """Normal-Inverse-Gamma-Normal metamodel for MyMetamodel.

    The metamodel is named ``nig_normal``.
    
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
    """

    def __init__(self,data,typespec): 
        super(NIGNormalMetamodel, self).__init__(data,typespec)
    
    def name(self): return 'nig_normal'
  
    def infer(self, threshold, numsamples=None):
        """Predict a value for each missing value in self.data."""
        nanrows, nancols = np.where(np.isnan(self.data))
        for i in numpy.arange(nancols.size):
            prediction = self.predict(nancols[i], threshold, numsamples)
            self.data[nanrows[i],nancols[i]] = prediction
    
    def predict(self, colno, threshold, numsamples=None):
        """Predict a value for a column, if confidence is high enough."""
        super(TrollMetamodel,self).predict(colno,threshold,numsamples=None)
    
    def predict_confidence(self, colno, numsamples=None):
        """Predict a value for a column and return confidence"""
        # value = 9.
        # confidence = 1.
        # return value, confidence
        raise NotImplementedError
        
    def simulate(self, colnos, constraints, numpredictions=1):
        """Simulate 'colnos' from a generator, subject to 'constraints'.
        
        Returns a list of rows with values for the specified columns.
        
        'colnos'         - list of column numbers.
        'constraints'    - list of ''(colno, value)'' pairs.
        'numpredictions' - number of results to return.
        """ 
        # check_col(colnos)
        # check_col([i for i,_ in constraints])
        # for (_, value) in constraints:
        #     if not value == 9:
        #         return float("nan")
        # header = [colnos]
        # X = numpy.array([[9. for _ in colnos] for _ in numpy.arange(numpredictions)])
        # return header, X
        raise NotImplementedError
    
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
        # for (_, value) in constraints:
        #     if not value == 9:
        #         return float("nan")
        # for (_, value) in targets:
        #     if not value == 9:
        #         return float("-inf")
            
        # '''Recheck this later'''
        # return 0
        raise NotImplementedError
    '''TODO: Rewrite or import docstrings'''