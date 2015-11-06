from MyMetamodel import *
import numpy as np
import numpy


class TrollMetamodel(MyMetamodel):
    """Troll metamodel for MyMetamodel.

    The metamodel is named ``troll_rng``.
    
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
        super(TrollMetamodel, self).__init__(data,typespec)
    
    def name(self): return 'troll_rng'
  
    def infer(self, threshold=0, numsamples=None):
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
        value = 9.
        confidence = float('inf')
        return value, confidence
        
    def simulate(self, colnos, constraints=[], numpredictions=1):
        # check_col(colnos)
        # check_col([i for i,_ in constraints])
        for (_, value) in constraints:
            if not value == 9:
                return float("nan")
        header = [colnos]
        X = np.array([[9. for _ in colnos] for _ in np.arange(numpredictions)])
        return header, X
    
    def estimate(self, colno0, colno1):
        # check_col(colno0, colno1)
        dep_prob = 0
        return dep_prob
    
    def column_mutual_information(self, colno0, colno1, numsamples=None):
        mutual_info = 0   
        return mutual_info
    
    def logpdf(self, targets, constraints):
        for (_, value) in constraints:
            if not value == 9:
                return float("nan")
        for (_, value) in targets:
            if not value == 9:
                return float("-inf")
            
        '''Recheck this later'''
        return 0
    
    '''TODO: Rewrite or import docstrings'''

X = np.tile(9., [10,5])
for [i,j] in [[0,0],[0,2],[1,3],[3,4],[4,4],[5,1]]:
    X[i][j] = float('nan')

troll = TrollMetamodel(X,'standard')


# print(troll.simulate([1,2,3]))

# print(troll.estimate(1,2))

# print(np.where(np.isnan(troll.data)))

troll.infer()
print(troll.data)

