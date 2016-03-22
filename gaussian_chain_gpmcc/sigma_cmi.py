import numpy as np
from numpy.linalg import det


def variance(node, DAG, noise, Sigma):
    inSigma = Sigma.copy()
    # Make NaN's of nodes without interaction equal zero
    inSigma[np.logical_not(np.outer(DAG[node]!=0,DAG[node]!=0))] = 0
    return DAG[node].dot(inSigma.dot(DAG[node].T)) + noise[node]

def covariance(target, parent, DAG, noise, Sigma):
    if parent in descendants(target,DAG):
        #Swap parent and target
        parent = parent + target
        target = parent - target
        parent = parent - target
        
    if parent==target:
        return variance(target, DAG, noise, Sigma)
    else:
        inSigma = Sigma.copy()
        # NaN's without interaction do not matter
        inSigma[parent,DAG[target]==0] = 0
        
        cov = DAG[target].dot(inSigma[parent].T)
        assert not np.isnan(cov),'Cov[%d,%d] is NaN'%(target,parent)
        return cov

def children(node,DAG):
    return np.nonzero(DAG[:,node])[0]

def parents(node,DAG):
    return np.nonzero(DAG[node,:])[0]
    
def ancestors(node, DAG):
    if (DAG[node]**2).sum() == 0:
        return np.array([])
    else:
        node_parents = parents(node,DAG)
        for i in node_parents:
            node_parents = np.append(node_parents, ancestors(i, DAG))
    return np.sort(np.unique(node_parents))

def descendants(node, DAG):
    return ancestors(node, DAG.T)

def roots(DAG):
    return np.where((DAG**2).sum(axis=1)==0)[0]

def update_roots(DAG, old_roots):
    if old_roots == np.array([]):
        return roots(DAG)
    else:
        inDAG = DAG.copy()
        inDAG[:,old_roots] = 0
        return roots(inDAG)
    
def isconnected(DAG):
    all_nodes = np.arange(0,DAG.shape[0])
    DAG_roots = roots(DAG)
    
    connected_nodes=DAG_roots
    for iRoot in DAG_roots:
        connected_nodes = np.append(connected_nodes,descendants(iRoot,DAG))
        
    return np.sort(np.unique(connected_nodes)).tolist() == all_nodes.tolist()

def automatic_Sigma (DAG, noise):
    '''Compute the covariance matrix Sigma of jointly gaussian distributed
    variables, additively related, analytically, given the DAG structure and
    independent (also gaussian) noise terms.
        DAG - NxN matrix, where N is the number of variables and DAG[i,j] is
            the coefficient, in the structural equation model, of variable j on i.  
        noise - Nx1 array with the variances of the independent gaussian noise
            for each variable
    ''' 
    Sigma = DAG*np.nan
    all_nodes = np.arange(0,DAG.shape[0])
    top_nodes = np.array([]).astype(int)
    
    assert isconnected(DAG),\
        "DAG is not connected"
    
    while top_nodes.tolist() != all_nodes.tolist():
        old_nodes = top_nodes
        top_nodes = update_roots(DAG,old_nodes)
        for iNode in top_nodes:
            for jNode in top_nodes[::-1]:
                if np.isnan(Sigma[iNode,jNode]):
                    Sigma[iNode,jNode] = covariance(iNode,jNode,DAG,noise,Sigma)
                    if np.isnan(Sigma[jNode,iNode]) and iNode!=jNode:
                        Sigma[jNode,iNode] = Sigma[iNode,jNode]
                    
    return Sigma

def submatrix(Sigma, index_list):
    if not index_list:
        return np.eye(2)
    else:
        return Sigma[np.ix_(index_list,index_list)]


def gaussian_cmi(Sigma,X,Y,givens):
    '''Compute cmi(X;Y|givens) for {X,Y,givens} jointly gaussian distributed
        Sigma - multivariate covariance matrix
        X - column index of first target
        Y - column index of second target
        givens - list of column indexes of variables to condition on
    '''
    assert type(givens)==list
    Sigma_Xgivens = submatrix(Sigma,[X] + givens)
    Sigma_Ygivens = submatrix(Sigma,[Y] + givens)
    Sigma_tot = submatrix(Sigma, [X,Y] + givens)
    Sigma_givens = submatrix(Sigma,givens)
    
    return .5*np.log( det(Sigma_Xgivens) * det(Sigma_Ygivens) / (det(Sigma_tot) * det(Sigma_givens) ))