from abc import ABCMeta, abstractmethod

def check_data(data):
    """Check whether data is of proper format (numpy array for now)"""
    print("Data check has not been implemented yet")
    
def check_typespec_dict(typespec):
    """Check whether typespec is a dict mapping variable names to type specs."""
    print("Typespec check has not been implemented yet")


class MyMetamodel(object):
    """Metamodel interface.
    
    Subclasses of :class:'MyMetamodel' implement the functionality needed to
    sample from and inquire about the posterior distribution of a generative
    model conditioned on data in a table.
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __init__ (self, data, typespec):
        check_data(data)
        check_typespec_dict(typespec)
        
        self.typespec = typespec
        self.data = data
        
    @abstractmethod
    def name(self):
        """Return the name of the metamodel as a str"""
        raise NotImplementedError
    
    @abstractmethod
    def infer(self):
        """Predict a value for each missing value."""
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, colno, threshold, numsamples=None):
        """Predict a value for a column, if confidence is high enough."""
        raise NotImplementedError
    
    @abstractmethod
    def predict_confidence(self, colno, numsamples=None):
        """Predict a value for a column and return confidence"""
        raise NotImplementedError
        
    @abstractmethod    
    def simulate(self, colnos, constraints, numpredictions=1):
        """Simulate 'colnos' from a generator, subject to 'constraints'.
        
        Returns a list of rows with values for the specified columns.
        
        'colnos'         - list of column numbers.
        'constraints'    - list of ''(colno, value)'' pairs.
        'numpredictions' - number of results to return.
        """ 
        raise NotImplementedError
    
    @abstractmethod
    def estimate(self, colno0, colno1):
        """Compute Dependence Probability of <col0> with <col1>"""
        raise NotImplementedError
    
    @abstractmethod
    def logpdf(self, target, constraints):
        """Compute (predictive?) logpdf.
        
                'constraints'    - list of ''(colno, value)'' pairs.
                'targets'        - list of ''(colno, value)'' pairs.
        """
        raise NotImplementedError
        
    @abstractmethod
    def column_mutual_information(self, colno0, colno1, numsamples=100):
        """Compute ``MUTUAL INFORMATION OF <col0> WITH <col1>``."""
        raise NotImplementedError