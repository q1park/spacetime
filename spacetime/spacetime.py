import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from spacetime.simulate import simulate_random_dag, simulate_ordered_dag, simulate_sem, simulate_data

class SpaceTime:
    def __init__(self, node_dict = dict(), features = list()):
        self.graph = nx.DiGraph(**{'features':features})
        self.order = node_dict
        self.data = None
        
        for t, time_slice in node_dict.items():
            for node in time_slice:
                self.add_node(node)
            
        self.time_order_nodes()
        
    def add_node(self, node, latent = False, pos = None, features = list()):
        pos = (np.random.uniform(0,1), np.random.uniform(0,1)) if type(pos) != tuple else pos
        self.graph.add_node(node, **{'latent':latent, 'features':features, 'pos':pos})
    
    def add_edge(self, src, dest, causal = True, features = list()):
        edge_type = '' if causal else 'X'
        self.graph.add_edge(src, dest, **{'features':features, 'causal':edge_type})
        
    def generate_random(self, variable_size, degree, graph_type, seed = 0):
        self.graph = simulate_random_dag(variable_size, degree, graph_type, seed = seed)
        self.infer_order()
        self.label_order_nodes()
    
    def generate_ordered(self, degree, graph_type, seed = 0):
        self.graph = simulate_ordered_dag(self.order, degree, graph_type, seed = seed)
    
    def generate_data(self, sample_size, x_dims, sem_type, linear_type, noise_scale = 1.0, seed = 0):
        self.data = simulate_sem(self.graph, sample_size, x_dims, sem_type, linear_type, 
                                 noise_scale = noise_scale, seed = seed)
            
    def label_order_nodes(self):
        mapping = dict()
        inode = 0
        for t, time_slice in self.order.items():
            for i, node in enumerate(time_slice):
                self.order[t][i] = inode
                mapping.update({node:inode})
                inode += 1
        self.graph = nx.relabel_nodes(self.graph, mapping)
        
    def time_order_nodes(self):
        for t, step in self.order.items():
            for x, node in enumerate(step):
                shift = np.random.uniform(0,0.2) if x%2==0 else 0
                shiftt = np.random.uniform(0,0.2) if t%2==0 else 0
                self.graph.nodes[node]['pos'] = (t+shift, x+shiftt)
    
    def connected_components(self, edges = 'all'):
        assert edges in ['all', 'directed', 'undirected']
        if edges == 'all':
            edge_list = self.graph.edges()
        elif edges == 'directed':
            edge_list = [x for x in self.graph.edges() if reversed(x) not in self.graph.edges()]
        else:
            edge_list = [x for x in self.graph.edges() if reversed(x) in self.graph.edges()]
            
        adjacencies = self._get_adj(edge_list)
        visited = {node:False for node in self.graph.nodes()} 
        
        components = list()  
        for node in self.graph.nodes(): 
            if visited[node] == False: 
                temp = list() 
                components.append(tuple(self._dfs_util(node, adjacencies, visited, temp))) 
        return components
    
    def infer_order(self):
        self.order.clear()
        ordered_nodes, unordered_nodes = set(), set(self.graph.nodes())
        components = self.connected_components(edges = 'undirected')
        
        itime = 0
        stop = False
        while not stop:
            time_slice = list()
            for comp in list(components):
                if not self._is_dest(comp, unordered_nodes-set(comp)):
                    if itime == 0:
                        time_slice += comp
                        components.remove(comp)
                    elif self._is_dest(comp, ordered_nodes):
                        time_slice += comp
                        components.remove(comp)
            if len(time_slice) > 0:
                self.order[itime] = time_slice
                ordered_nodes.update(self.order[itime])
                unordered_nodes = unordered_nodes - ordered_nodes
                itime += 1
            else:
                stop = True
        self.time_order_nodes()
    
    def is_causal(self, src, dest):
        seen = [src]
        neighbors = [(src,neighbor) for neighbor in self.graph.neighbors(src)]
        
        while neighbors:
            (s, d) = neighbors.pop()
            if d in self.graph[s] and self.graph[s][d]['causal']=='':
                if d == dest:
                    return True
                if d not in seen:
                    neighbors += [(d, neighbor) for neighbor in self.graph.neighbors(d)]
                seen.append(d)
        return False
    
    def _dfs_util(self, node, adjacencies, visited, temp): 
        visited[node] = True
        temp += [node]
        for node in adjacencies[node]: 
            if visited[node] == False: 
                  temp = self._dfs_util(node, adjacencies, visited, temp) 
        return temp 
    
    def _get_adj(self, edge_list):
        adj_dict = {node:set() for node in self.graph.nodes()}
        for src, dest in edge_list:
            adj_dict[src].add(dest)
        return adj_dict
    
    def _is_dest(self, dest_nodes, src_nodes):
        for src in src_nodes:
            for dest in dest_nodes:
                if dest in self.graph[src]:
                    return True
        return False
    
    def ordered_adj(self):
        return np.around(nx.to_numpy_array(self.graph, nodelist=sorted(self.graph._node.keys())), 3)
    
    def draw_graph(self):
        nx.draw_networkx(self.graph, with_labels=True, 
                         pos=nx.get_node_attributes(self.graph,'pos'),
                         node_size=800, node_color='gray')
        nx.draw_networkx_edge_labels(self.graph, 
                                     pos=nx.get_node_attributes(self.graph,'pos'),
                                     edge_labels=nx.get_edge_attributes(self.graph,'causal'))
        plt.axis('off')
        plt.show()    