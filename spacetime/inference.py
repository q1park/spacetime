import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats
import itertools
import networkx as nx
from collections import Counter

class Infer:
    def __init__(self, spacetime, data, algorithm, dseparation_test):
        self.st = spacetime
        self.data = data
        self.algorithm = algorithm
        self.dseparation_test = dseparation_test
        self.sepset = dict()
        
        assert len(data.columns) == len(self.st.graph.nodes())
        for node in self.st.graph.nodes():
            assert node in data.columns
                
    def initialize_graph(self, timing = False):
        t_past = None
        for t, step in self.st.order.items():
            if not timing:                    
                for (src, dest) in itertools.permutations(step, 2):
                    self.st.add_edge(src, dest, causal=False)

            if t_past is not None:
                for (src, dest) in itertools.product(self.st.order[t_past], self.st.order[t]):
                    self.st.add_edge(src, dest, causal=False)
            t_past = t
            
    def d_separate(self, alpha=0.05, max_k=None):
        ### Test for d-separation x - sepset - y
        self.sepset = {x:set() 
                       for x in sorted([tuple(sorted(pair)) 
                                        for pair in itertools.combinations(self.st.graph.nodes(), 2)])}
        max_k = len(self.st.graph.nodes)+1 if max_k is None else max_k
        for N in range(max_k + 1):
            for (x, y) in self.sepset.keys():
                x_neighbors = list(set(nx.all_neighbors(self.st.graph, x)))
                y_neighbors = list(set(nx.all_neighbors(self.st.graph, y)))
                z_candidates = list(set(x_neighbors + y_neighbors) - set([x,y]))
                
                for z in itertools.combinations(z_candidates, N):
                    test = self.dseparation_test([y], [x], list(z), self.data, alpha)
                    if test.independent():
                        if self.st.graph.has_edge(x,y):
                            self.st.graph.remove_edge(x,y)
                            self.sepset[(x,y)].update(z)
                        if self.st.graph.has_edge(y,x):
                            self.st.graph.remove_edge(y,x)
                            self.sepset[(x,y)].update(z)
                        break
                        
        ### Search for collider nodes x > z < y
        for z in self.st.graph.nodes():
            for (x,y) in itertools.combinations(set(nx.all_neighbors(self.st.graph, z)), 2):
                if self.st.graph.has_edge(x,y) or self.st.graph.has_edge(y,x):
                    continue
                if z not in self.sepset[tuple(sorted((x,y)))]:
                    if self.st.graph.has_edge(z,x):
                        self.st.graph.remove_edge(z,x)
                    if self.st.graph.has_edge(z,y):
                        self.st.graph.remove_edge(z,y)
                        
    def infer_latent(self):
        added_arrows = True
        
        while added_arrows:
            added_arrows_1 = self._recursion_1()
            added_arrows_2 = self._recursion_2()
            added_arrows = added_arrows_1 or added_arrows_2
            
    def model_latent(self):
        latent_adjs = list()

        for t, step in self.st.order.copy().items():
            t_latent = t-1
            for node in step:
                for edge in list(self.st.graph.in_edges(node))+list(self.st.graph.out_edges(node)):
                    adj = tuple(sorted(edge))
                    causal = True if self.st.graph.get_edge_data(*edge)['causal']=='' else False

                    if adj in latent_adjs:
                        self.st.graph[edge[0]][edge[1]]['causal'] = ''
                        continue

                    if not causal:
                        if t_latent not in self.st.order.keys():
                            self.st.order[t_latent] = list()

                        latent_name = 'L_%s_%s'%(adj[0], adj[1])
                        self.st.order[t_latent] += [latent_name]
                        self.st.add_node(latent_name, latent=True)
                        self.st.add_edge(latent_name, adj[0])
                        self.st.add_edge(latent_name, adj[1])
                        self.st.graph[edge[0]][edge[1]]['causal'] = ''

                    latent_adjs += [adj]
        self.st.time_order_nodes()

    
    def _recursion_1(self):
        added_arrows = False
        for c in self.st.graph.nodes():
            for (a,b) in itertools.combinations(set(nx.all_neighbors(self.st.graph, c)), 2):
                if self.st.graph.has_edge(a,b) or self.st.graph.has_edge(b,a):
                    continue
                if a not in self.st.graph[c] and b in self.st.graph[c] and not self.st.graph[c][b]['causal']=='':
                    if c in self.st.graph[b]:
                        self.st.graph.remove_edge(b,c)
                    self.st.graph[c][b]['causal']=''
                    added_arrows = True
                if b not in self.st.graph[c] and a in self.st.graph[c] and not self.st.graph[c][a]['causal']=='':
                    if c in self.st.graph[a]:
                        self.st.graph.remove_edge(a,c)
                    self.st.graph[c][a]['causal']=''
                    added_arrows = True
        return added_arrows

    def _recursion_2(self):
        added_arrows = False
        for (a,b) in self.st.graph.edges():
            if self.st.is_causal(b,a):
                self.st.graph.remove_edge(a,b)
        return added_arrows

DEFAULT_BINS = 2

class RobustRegressionTest():
    def __init__(self, y, x, z, data, alpha):
        self.regression = sm.RLM(data[y], data[x+z])
        self.result = self.regression.fit()

        self.coefficient = self.result.params[x].iloc[0]
        confidence_interval = self.result.conf_int(alpha=alpha/2.)
        self.upper = confidence_interval[1][x].iloc[0]
        self.lower = confidence_interval[0][x].iloc[0]

    def independent(self):
        if self.coefficient > 0.:
            if self.lower > 0.:
                return False
            else:
                return True
        else:
            if self.upper < 0.:
                return False
            else:
                return True

class MutualInformationTest():
    """
    This is mostly from "Distribution of Mutual Information" by Marcus Hutter.  This MVP implementation
    doesn't contain priors, but will soon be adjusted to include the priors for n_xy.

    It uses a very basic variance estimate on MI to get approximate confidence intervals
    on I(X,Y|Z=z) for each z, then basic error propagation (incorrectly assuming 0 covariance, i.e.
    Cov(I(X,Y|Z=z_i), I(X,Y|Z=z_j)) = 0.  This second assumption results in an underestimate of the
    final confidence interval.
    """
    def __init__(self, y, x, z, X, alpha, variable_types={}):
        self.I, self.dI = self.discrete_mutual_information(x, y, z, X)
        z = scipy.stats.norm.ppf(1.-alpha/2.) # one-sided
        self.dI = z*self.dI

    def independent(self):
        if self.I - self.dI > 0.:
            return False
        else:
            return True

    def discrete_mutual_information(self, x, y, z, X):
        n_z = Counter()
        for zi in X[z].values:
            n_z[tuple(zi)] += 1.
        N = sum(n_z.values())
        conditional_informations = {}
        for zi, n_zi in n_z.items():
            zi_subset = X.copy()
            for col, val in zip(z,zi):
                zi_subset = zi_subset[zi_subset[col] == val]
            conditional_informations[zi] = self.max_likelihood_information(x,y,zi_subset)
        I_ml = sum([(kz/N)*conditional_informations[zi][0] for zi, kz in n_z.items()])
        dI_ml = np.sqrt(sum([((kz/N)*conditional_informations[zi][1])**2. for zi, kz in n_z.items()]))
        return I_ml, dI_ml

    def max_likelihood_information(self, x, y, X):
        """
        This estimator appears to get very imprecise quickly as the dimensions and
        cardinality of x and y get larger.  It works well for dimensions around 1,
        and cardinality around 5.  Higher dimensions require lower cardinality.  For
        further refinment, I'll have to see if using a prior for I(x,y) helps.
        """
        n_x, n_y, n_xy = Counter(), Counter(), Counter()

        for xy in X[x+y].values:
            xi = xy[:len(x)]
            yi = xy[len(x):]
            n_x[tuple(xi)] += 1.
            n_y[tuple(yi)] += 1.
            n_xy[(tuple(xi),tuple(yi))] += 1.
        N = sum(n_x.values())
        I_ml = sum([(k / N) * np.log(k * N / float(n_x[xi]*n_y[yi])) for (xi,yi), k in n_xy.items()])
        K = sum([(k / N) * (np.log(k * N / float(n_x[xi]*n_y[yi])))**2. for (xi,yi), k in n_xy.items()])
        return I_ml, np.sqrt((K - I_ml**2.)/(N + 1.))