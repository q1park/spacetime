import time
import numpy as np
import networkx as nx
from statsmodels.nonparametric.kernel_density import KDEMultivariate, KDEMultivariateConditional, EstimatorSettings
from spacetime.spacetime import NodeData

class Histogram:
    def __init__(self, node_data, epsilon = 1e-15):
        self.nodes = node_data.nodes
        self.joint = self._joint_full(node_data.data(), bins=node_data.edges())
        self.epsilon = epsilon
        
    def _joint_full(self, data, bins):
        joint, bnds = np.histogramdd(np.squeeze(data), bins=bins)
        
        if np.sum(joint)==0.0:
            raise ValueError("zero joint distribution")
            
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
        
    def _compute_joint_kde(self, *nodes, normref=True):
        endog = [self.node_data.info[node]['data'] for node in nodes]
        t = time.time()
        if normref:
            kde = KDEMultivariate(data=endog, var_type='c'*len(nodes), 
                                  bw='normal_reference')
        else:
            kde = KDEMultivariate(data=endog, var_type='c'*len(nodes), 
                                  bw='cv_ml', defaults=EstimatorSettings(efficient=True))
        print("Fit joint KDE for %s in %s seconds"%(nodes, time.time() - t))
        self.kdes_joint[nodes] = kde
        
    def _compute_conditional_kde(self, dep, inds, normref=True):
        endog = self.node_data.info[dep]['data']
        exog = [self.node_data.info[node]['data'] for node in inds]
        t = time.time()
        if normref:
            kde = KDEMultivariateConditional(endog=endog, exog=exog, 
                                             dep_type='c', indep_type='c'*len(exog), 
                                             bw='normal_reference')
        else:
            kde = KDEMultivariateConditional(endog=endog, exog=exog, 
                                             dep_type='c', indep_type='c'*len(exog), 
                                             bw='cv_ml', defaults=EstimatorSettings(efficient=True))
        print("Fit conditional KDE for %s wrt %s in %s seconds"%(dep, inds, time.time() - t))
        self.kdes_conditional[dep][inds] = kde
            
    def compute_joint(self, *nodes):
        if nodes not in self.kdes_joint.keys():
            self._compute_joint_kde(*nodes)
        
        kde = self.kdes_joint[nodes]
        distro = np.zeros([len(self.node_data.info[node]['axis']) for node in nodes])
        
        bins, points = self.node_data.product_idxs(*nodes), self.node_data.product_axes(*nodes)
        
        t = time.time()
        for b, p in zip(bins, points):
            b = b.reshape(len(nodes),-1)
            distro[np.ix_(*b)] = kde.pdf(data_predict=p)
                
        distro /= np.sum(distro)
#         print("Computed joint distro for %s in %s seconds"%(nodes, time.time() - t))
        return nodes, distro
        
    def compute_conditional(self, dep, inds, normref=True):
        nodes = tuple([dep]) + inds
        
        if inds not in self.kdes_conditional[dep].keys():
            self._compute_conditional_kde(dep, inds, normref=normref)
            
        kde = self.kdes_conditional[dep][inds]
        distro = np.zeros([len(self.node_data.info[node]['axis']) for node in nodes])
  
        endobins, endopoints = self.node_data.product_idxs(dep), self.node_data.product_axes(dep)
        exobins, exopoints = self.node_data.product_idxs(*inds), self.node_data.product_axes(*inds)
        
        t = time.time()
        for exob, exop in zip(exobins, exopoints):
            for endob, endop in zip(endobins, endopoints):
                b = np.concatenate([endob, exob]).reshape(len(nodes),-1)
                distro[np.ix_(*b)] = kde.pdf(endog_predict=endop, exog_predict=exop)

            slices = tuple([slice(None)]+[slice(i,i+1,1) for i in exob])
            distro[slices] /= np.sum(distro[slices])
#         print("Computed conditional distro for %s wrt %s in %s seconds"%(dep, inds, time.time() - t))
        return nodes, distro
    
class PDFVisualizer:
    def __init__(self, spacetime, node_data, observables = ()):
        assert spacetime.A.shape[0]==len(node_data.nodes)
        
        self.st = spacetime
        self.node_data = node_data
        
        if len(observables)>0:
            obs = list(observables)
            _node_data, _bins = self.node_data.data()[:,obs,:], self.node_data.edges(*obs)
            self.histogram = Histogram(NodeData(_node_data, bins=_bins))
            self.kde = KDE(NodeData(_node_data, bins=_bins))
            self.histo_dict = dict(zip(obs, range(len(obs))))
        else:
            obs = list(self.node_data.nodes)
            self.histogram = Histogram(self.node_data)
            self.kde = KDE(self.node_data)
            self.histo_dict = dict(zip(obs, range(len(obs))))
        self.histo_dict_inv = {v:k for k,v in self.histo_dict.items()}
    
    def get_plot_marginal(self, var):
        idx = self.histo_dict[var]
        nodes, joint = self.histogram.compute_joint(idx)
        nodes = [self.histo_dict_inv[x] for x in nodes]
        axes = self.node_data.axes(*nodes)
        return axes[0], joint.T
    
    def get_contour_joint(self, var_x, var_y, smooth=False):
        idx, idy = self.histo_dict[var_x], self.histo_dict[var_y]
        if smooth:
            nodes, joint = self.kde.compute_joint(idx, idy)
        else:
            nodes, joint = self.histogram.compute_joint(idx, idy)
        nodes = [self.histo_dict_inv[x] for x in nodes]
        axes = self.node_data.axes(*nodes)
        mesh = np.meshgrid(*axes)
        return mesh[0], mesh[1], joint.T
    
    def get_contour_conditional(self, dep, ind, smooth=False, normref=True):
        iddep, idind = self.histo_dict[dep], self.histo_dict[ind]
        if smooth:
            nodes, conditional = self.kde.compute_conditional(iddep, tuple([idind]), normref=normref)
        else:
            nodes, conditional = self.histogram.compute_conditional(iddep, tuple([idind]))
        nodes = [self.histo_dict_inv[x] for x in nodes]
        axes = self.node_data.axes(*nodes)
        mesh = np.meshgrid(*axes)
        return mesh[0], mesh[1], conditional.T
    
class SCalculator:
    def __init__(self, node_dataP, node_dataQ, observables = ()):
        assert len(node_dataP.edges()) == len(node_dataQ.edges())
        
        for axP, axQ in zip(node_dataP.edges(), node_dataQ.edges()):
            assert len(axP) == len(axQ)
            assert all([axP[i]==axQ[i] for i in range(len(axP))])
            
        self.node_dataP = node_dataP
        self.node_dataQ = node_dataQ
        
        if len(observables)>0:
            obs = list(observables)
            _node_dataP, _binsP = self.node_dataP.data()[:,obs,:], self.node_dataP.edges(*obs)
            _node_dataQ, _binsQ = self.node_dataQ.data()[:,obs,:], self.node_dataQ.edges(*obs)
            self.histogramP = Histogram(NodeData(_node_dataP, bins=_binsP))
            self.histogramQ = Histogram(NodeData(_node_dataQ, bins=_binsQ))
            self.kdeP = KDE(NodeData(_node_dataP, bins=_binsP))
            self.kdeQ = KDE(NodeData(_node_dataQ, bins=_binsQ))
            self.histo_dict = dict(zip(obs, range(len(obs))))
        else:
            obs = list(self.node_dataP.nodes)
            self.histogramP = Histogram(self.node_dataP)
            self.histogramQ = Histogram(self.node_dataQ)
            self.kdeP = Histogram(self.node_dataP)
            self.kdeQ = Histogram(self.node_dataQ)
            self.histo_dict = dict(zip(obs, range(len(obs))))
        self.histo_dict_inv = {v:k for k,v in self.histo_dict.items()}
        
    def get_conditional(self, dep, ind, smooth):
        iddep, idind = self.histo_dict[dep], self.histo_dict[ind]
        nodes, conditionalP = self.kdeP.compute_conditional(iddep, tuple([idind]))
        if smooth:
#             nodes, conditionalP = self.kdeP.compute_conditional(iddep, tuple([idind]))
            nodes, conditionalQ = self.kdeQ.compute_conditional(iddep, tuple([idind]))
        else:
#             nodes, conditionalP = self.histogramP.compute_conditional(iddep, tuple([idind]))
            nodes, conditionalQ = self.histogramQ.compute_conditional(iddep, tuple([idind]))
        nodes = [self.histo_dict_inv[x] for x in nodes]
        axis = self.node_dataP.axes(*nodes)[1]
        return axis, conditionalP, conditionalQ
    
    def compute_S(self, dep, ind, smooth=True):
        ax, P, Q = self.get_conditional(dep, ind, smooth=smooth)
        return ax, np.array([SCalculator.S(P[:,i], Q[:,i]) for i in range(len(ax))])
    
    @staticmethod
    def S(P, Q):
        eps = 1e-15
        return np.sum(P*np.log(eps+P/(Q+eps)))