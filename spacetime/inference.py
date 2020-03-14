import itertools
import networkx as nx

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