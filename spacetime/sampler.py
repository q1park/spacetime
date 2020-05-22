import time
import numpy as np
import networkx as nx
from statsmodels.nonparametric.kernel_density import KDEMultivariate, KDEMultivariateConditional, EstimatorSettings

class NodeData:
    def __init__(self, data, bins):
        self.nodes = tuple(range(data.shape[1]))
        self.info = {node:self.bin_data(node, data, bins) for node in self.nodes}
    
    def bin_data(self, node, data, bins):
        if type(bins)==list:
            edges = bins[node]
        elif type(bins)==int:
            data_min, data_max = data[:,node,:].min(), data[:,node,:].max()
            edges = np.linspace(*self.make_bins(data_min, data_max), bins+1+node)
        else:
            raise TypeError("bins must be type int or list")
            
        axis = edges[:-1]+np.diff(edges)/2
        return {'data':data[:,node,:], 'edges':edges, 'axis':axis}
    
    def make_bins(self, data_min, data_max):
        round_bins = lambda x: np.around(x, -int(np.floor(np.log10(np.abs(x))))+2)
        bin_min = data_min-0.25*(data_max-data_min)
        bin_max = data_max+0.25*(data_max-data_min)
        return round_bins(bin_min), round_bins(bin_max)
    
    def data(self):
        data_list = [info['data'] for node, info in self.info.items()]
        return np.expand_dims(np.hstack(data_list), axis = 2)
    
    def axes(self, *nodes):
        nodes = self.nodes if len(nodes) == 0 else nodes
        return [self.info[node]['axis'] for node in nodes]
    
    def idxs(self, *nodes):
        nodes = self.nodes if len(nodes) == 0 else nodes
        return [range(len(self.info[node]['axis'])) for node in nodes]
    
    def edges(self, *nodes):
        nodes = self.nodes if len(nodes) == 0 else nodes
        return [self.info[node]['edges'] for node in nodes] 
    
    def product_axes(self, *nodes):
        return np.array(np.meshgrid(*self.axes(*nodes))).T.reshape(-1,len(nodes))
    
    def product_idxs(self, *nodes):
        return np.array(np.meshgrid(*self.idxs(*nodes))).T.reshape(-1,len(nodes))


class GraphSampler:
    def __init__(self, spacetime, node_data):
        assert spacetime.adj.shape[0]==len(node_data.nodes)
        
        self.st = spacetime
        self.node_data = node_data
        
        self.histogram = Histogram(self.node_data)
        self.kde = KDE(self.node_data)
        
        self.node_cdfs = {node:dict() for node in self.node_data.nodes}
        self.node_pdfs = {node:dict() for node in self.node_data.nodes}
    
    def _compute_from_kdes(self, htype):
        G = nx.DiGraph(self.st.adj.numpy())
        
        for node in self.node_data.nodes:
            inds = tuple(G.predecessors(node))

            if len(inds)==0:
                nodes, distro = self.kde.compute_joint(node, htype=htype)
            else:
                nodes, distro = self.kde.compute_conditional(node, inds, htype=htype)
                
            if htype=='pdf':
                self.node_pdfs[node][inds] = distro
            else:
                self.node_cdfs[node][inds] = distro
            
    def compute_mean_std(self, prob, axes):
        mean = prob.dot(axes)
        std = np.sqrt(prob.dot((mean-axes)**2))
        return mean, std

    def _sample_pdf(self, axis, pdf, size):
        mean, std = self.compute_mean_std(pdf, axis)
        return np.random.normal(mean, std, size=size)
    
    def _sample_cdf(self, axis, cdf, size):
        randv = np.random.uniform(size=size)
        # interpolation steps
        idx1 = np.clip(np.searchsorted(cdf, randv), 1, len(cdf))
        idx0 = np.where(idx1==0, 0, idx1-1)
        # linear interpolation
        frac = (randv-cdf[idx0])/(cdf[idx1]-cdf[idx0])
        return axis[idx0]*(1-frac) + axis[idx1]*frac
    
    def resample_from_pdf(self, size, mutilate=list()):
        resampled = {node:np.zeros(size) for node in self.node_data.nodes}
        if len(self.node_pdfs[self.node_data.nodes[-1]])==0:
            self._compute_from_kdes(htype='pdf')
        
        for endo, pdf_dict in self.node_pdfs.items():
            for exo, pdf in pdf_dict.items():
                if len(exo)==0:
                    resampled[endo] += self._sample_pdf(self.node_data.axes(endo)[0], 
                                                        pdf, size=size)
                elif endo in mutilate:
                    nodes, pdf = self.kde.compute_joint(endo, htype='pdf')
                    resampled[endo] += self._sample_pdf(self.node_data.axes(endo)[0], 
                                                        pdf, size=size)
                else:
                    pa_vals = [np.searchsorted(self.node_data.edges(node)[0][1:], 
                                               resampled[node]) for node in exo]

                    for pa_bins in self.node_data.product_idxs(*exo):
                        idxs = np.logical_and.reduce([x==idx for x, idx in zip(pa_vals, pa_bins)]).nonzero()[0]
                        slices = tuple([slice(None)]+[slice(i,i+1,1) for i in pa_bins])
                        resampled[endo][idxs] += self._sample_pdf(self.node_data.axes(endo)[0], 
                                                                  np.squeeze(pdf[slices]), 
                                                                  size=len(idxs))
        return np.expand_dims(np.hstack([d.reshape(size,-1) for d in resampled.values()]), axis = 2)
    
    def resample_from_cdf(self, size, mutilate=list()):
        resampled = {node:np.zeros(size) for node in self.node_data.nodes}
        if len(self.node_cdfs[self.node_data.nodes[-1]])==0:
            self._compute_from_kdes(htype='cdf')
        
        for endo, cdf_dict in self.node_cdfs.items():
            for exo, cdf in cdf_dict.items():
                if len(exo)==0:
                    resampled[endo] += self._sample_cdf(self.node_data.axes(endo)[0], 
                                                        cdf, size=size)
                elif endo in mutilate:
                    nodes, cdf = self.kde.compute_joint(endo, htype='cdf')
                    resampled[endo] += self._sample_cdf(self.node_data.axes(endo)[0], 
                                                        cdf, size=size)
                else:
                    pa_vals = [np.searchsorted(self.node_data.edges(node)[0][1:], 
                                               resampled[node]) for node in exo]

                    for pa_bins in self.node_data.product_idxs(*exo):
                        idxs = np.logical_and.reduce([x==idx for x, idx in zip(pa_vals, pa_bins)]).nonzero()[0]
                        slices = tuple([slice(None)]+[slice(i,i+1,1) for i in pa_bins])
                        resampled[endo][idxs] += self._sample_cdf(self.node_data.axes(endo)[0], 
                                                                  np.squeeze(cdf[slices]), 
                                                                  size=len(idxs))
        return np.expand_dims(np.hstack([d.reshape(size,-1) for d in resampled.values()]), axis = 2)
    
    def get_contour_joint(self, var_x, var_y):
        nodes, joint = self.histogram.compute_joint(var_x, var_y)
        axes = self.node_data.axes(*nodes)
        mesh = np.meshgrid(*axes)
        return mesh[0], mesh[1], joint.T
    
    def get_contour_conditional(self, dep, ind):
        nodes, conditional = self.histogram.compute_conditional(dep, tuple([ind]))
        axes = self.node_data.axes(*nodes)
        mesh = np.meshgrid(*axes)
        return mesh[0], mesh[1], conditional.T
    
    
class Histogram:
    def __init__(self, node_data, epsilon = 10e-10):
        self.nodes = node_data.nodes
        self.joint = self._joint_full(node_data.data(), bins=node_data.edges())
        self.epsilon = epsilon
        
    def _joint_full(self, data, bins):
        joint, bnds = np.histogramdd(np.squeeze(data), bins=bins)
        joint /= np.sum(joint)
        return joint
    
    def _integrate_out(self, *out_nodes, keepdims=False):
        in_nodes = tuple([node for node in self.nodes if node not in out_nodes])
        out_idxs = [i for i, node in enumerate(self.nodes) if node in out_nodes]
        return in_nodes, np.sum(self.joint, axis=tuple(out_idxs), keepdims=keepdims)
    
    def _reorder(self, new_order, old_order, data):
        perm = [old_order.index(x) for x in new_order]
        return new_order, np.transpose(data, tuple(perm))
    
    def _epsilonize(self, nodes, np_arr):
        np_arr[np_arr==0] = self.epsilon
        return nodes, np_arr
    
    def compute_joint(self, *nodes):
        out_nodes = [node for i, node in enumerate(self.nodes) if node not in nodes]
        return self._reorder(nodes, *self._integrate_out(*out_nodes))
    
    def compute_conditional(self, dep, inds):
        nodes = tuple([dep]) + inds
        nodes_out = tuple([node for node in self.nodes if node not in nodes])
        deps_out = tuple([dep]) + nodes_out
        
        old_nodes, num = self._integrate_out(*nodes_out, keepdims=True)
        _, den = self._epsilonize(*self._integrate_out(*deps_out, keepdims=True))
        
        return self._reorder(nodes, old_nodes, np.squeeze(num/den)) 
    

class KDE:
    def __init__(self, node_data):
        self.node_data = node_data
        self.nodes = node_data.nodes
        
        self.kdes_joint = dict()
        self.kdes_conditional = {node:dict() for node in self.nodes}
        self.distribution_types = ('pdf', 'cdf')
        
    def _compute_joint_kde(self, *nodes):
        endog = [self.node_data.info[node]['data'] for node in nodes]
        t = time.time()
        kde = KDEMultivariate(data=endog, var_type='c'*len(nodes), 
                              bw='normal_reference')
                              #bw='cv_ml', defaults=EstimatorSettings(efficient=True))
        print("Fit joint KDE for %s in %s seconds"%(nodes, time.time() - t))
        self.kdes_joint[nodes] = kde
        
    def _compute_conditional_kde(self, dep, inds):
        endog = self.node_data.info[dep]['data']
        exog = [self.node_data.info[node]['data'] for node in inds]
        t = time.time()
        kde = KDEMultivariateConditional(endog=endog, exog=exog, 
                                         dep_type='c', indep_type='c'*len(exog), 
                                         bw='normal_reference')
                                         #bw='cv_ml', defaults=EstimatorSettings(efficient=True))
        print("Fit conditional KDE for %s wrt %s in %s seconds"%(dep, inds, time.time() - t))
        self.kdes_conditional[dep][inds] = kde
            
    def compute_joint(self, *nodes, htype='pdf'):
        if htype not in self.distribution_types:
            raise ValueError("histo type must be %s or %s")%tuple(*self.distribution_types)
            
        if nodes not in self.kdes_joint.keys():
            self._compute_joint_kde(*nodes)
        
        kde = self.kdes_joint[nodes]
        distro = np.zeros([len(self.node_data.info[node]['axis']) for node in nodes])
        
        bins, points = self.node_data.product_idxs(*nodes), self.node_data.product_axes(*nodes)
        
        t = time.time()
        for b, p in zip(bins, points):
            b = b.reshape(len(nodes),-1)
            
            if htype=='pdf':
                distro[np.ix_(*b)] = kde.pdf(data_predict=p)
            else:
                distro[np.ix_(*b)] = kde.cdf(data_predict=p)
                
        if htype=='pdf':
            distro /= np.sum(distro)
        else:
            if len(nodes)==1:
                distro /= distro[-1]
            else:
                print("Returning un-normalized multi-dimensional cdf")
        print("Computed joint distro for %s in %s seconds"%(nodes, time.time() - t))
        return nodes, distro
        
    def compute_conditional(self, dep, inds, htype='pdf'):
        nodes = tuple([dep]) + inds
        if htype not in self.distribution_types:
            raise ValueError("histo type must be %s or %s")%tuple(*self.distribution_types)
        
        if inds not in self.kdes_conditional[dep].keys():
            self._compute_conditional_kde(dep, inds)
            
        kde = self.kdes_conditional[dep][inds]
        distro = np.zeros([len(self.node_data.info[node]['axis']) for node in nodes])
  
        endobins, endopoints = self.node_data.product_idxs(dep), self.node_data.product_axes(dep)
        exobins, exopoints = self.node_data.product_idxs(*inds), self.node_data.product_axes(*inds)
        
        t = time.time()
        for exob, exop in zip(exobins, exopoints):
            for endob, endop in zip(endobins, endopoints):
                b = np.concatenate([endob, exob]).reshape(len(nodes),-1)
                if htype=='pdf':
                    distro[np.ix_(*b)] = kde.pdf(endog_predict=endop, exog_predict=exop)
                else:
                    distro[np.ix_(*b)] = kde.cdf(endog_predict=endop, exog_predict=exop)

            slices = tuple([slice(None)]+[slice(i,i+1,1) for i in exob])

            if htype=='pdf':
                distro[slices] /= np.sum(distro[slices])
            else:
                distro[slices] /= distro[slices][-1]
        print("Computed conditional distro for %s wrt %s in %s seconds"%(dep, inds, time.time() - t))
        return nodes, distro