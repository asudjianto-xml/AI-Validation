"""Exact depth-two regression-tree tables with explicitly weighted purification.

Informed by Modeva's constant-tree extraction/mass-transfer implementation and
Lengerich et al. (2020). No Modeva imports or threshold perturbations are used.
Tree paths, not whole-tree feature sets, determine the maximum interaction order.
"""
import copy
import json
import numpy as np


def purify_pair(values, weights, tolerance=1e-10, max_iter=20000):
    """Transfer weighted row/column means into additive effects, preserving all cells.

    Empty rows/columns transfer zero. Zero-mass cells have no orthogonality claim.
    Failure to converge is an error, not a silently accepted decomposition.
    """
    v=np.asarray(values,float).copy(); w=np.asarray(weights,float)
    if v.shape!=w.shape or v.ndim!=2 or not np.isfinite(v).all():
        raise ValueError('Finite pair table and matching weights required')
    if not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Nonnegative finite weights with positive mass required')
    w=w/w.sum(); wr=w.sum(1); wc=w.sum(0)
    a=np.zeros(v.shape[0]); b=np.zeros(v.shape[1])
    for iteration in range(max_iter):
        row=np.divide((w*v).sum(1),wr,out=np.zeros_like(wr),where=wr>0)
        a+=row; v-=row[:,None]
        col=np.divide((w*v).sum(0),wc,out=np.zeros_like(wc),where=wc>0)
        b+=col; v-=col[None,:]
        r=np.divide((w*v).sum(1),wr,out=np.zeros_like(wr),where=wr>0)
        c=np.divide((w*v).sum(0),wc,out=np.zeros_like(wc),where=wc>0)
        error=float(max(np.max(abs(r)),np.max(abs(c))))
        if error<tolerance:
            np.testing.assert_allclose(v+a[:,None]+b[None,:],values,atol=1e-11,rtol=1e-11)
            return v,a,b,dict(iterations=iteration+1,max_conditional_mean=error)
    raise RuntimeError(f'Purification failed to converge: {error}')


class DepthTwoFANOVA:
    def __init__(self, model, X_reference, shared_cuts=None):
        X=np.asarray(X_reference,dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError('Missing/nonfinite features require an explicit missing-value representation')
        self.n_features=X.shape[1]
        booster=model.get_booster(); names=booster.feature_names
        config=json.loads(booster.save_config())['learner']
        objective=config['objective']['name']
        if objective not in ('reg:squarederror','binary:logistic'):
            raise ValueError('Only scalar regression or binary logistic raw margins are supported')
        self.output_scale='logit' if objective=='binary:logistic' else 'identity'
        if config['gradient_booster']['name']!='gbtree':
            raise ValueError('Only ordinary gbtree is supported')
        base=config['learner_model_param']['base_score']
        self.intercept=float(base.strip('[]'))
        if objective=='binary:logistic':
            self.intercept=float(np.log(self.intercept/(1-self.intercept)))
        cuts=[set() for _ in range(self.n_features)]; leaves=[]
        def walk(node,bounds,depth):
            if 'leaf' in node:
                if len(bounds)>2 or depth>2:
                    raise ValueError('Only trees of depth at most two are supported')
                leaves.append((bounds,float(node['leaf']))); return
            key=node['split']; j=names.index(key) if names else int(key[1:])
            t=float(np.float32(node['split_condition'])); cuts[j].add(t)
            children={c['nodeid']:c for c in node['children']}
            lo,hi=bounds.get(j,(-np.inf,np.inf))
            left=dict(bounds);left[j]=(lo,min(hi,t))
            right=dict(bounds);right[j]=(max(lo,t),hi)
            walk(children[node['yes']],left,depth+1)
            walk(children[node['no']],right,depth+1)
        for dump in booster.get_dump(dump_format='json'):
            walk(json.loads(dump),{},0)
        if shared_cuts is not None:
            if len(shared_cuts)!=self.n_features:raise ValueError('Shared-grid dimension mismatch')
            for j,c in enumerate(cuts):
                if not c.issubset(set(np.asarray(shared_cuts[j],float))):
                    raise ValueError('Shared grids must include every fitted threshold')
            self.cuts=[np.asarray(c,float) for c in shared_cuts]
        else:
            self.cuts=[np.array(sorted(c),float) for c in cuts]
        self.main=[np.zeros(len(c)+1) for c in self.cuts];self.pair={}
        for bounds,value in leaves:
            dims=tuple(sorted(bounds))
            slices=[]
            for j in dims:
                lo,hi=bounds[j]
                a=0 if np.isneginf(lo) else np.searchsorted(self.cuts[j],lo)+1
                b=len(self.cuts[j])+1 if np.isposinf(hi) else np.searchsorted(self.cuts[j],hi)+1
                slices.append(slice(a,b))
            if not dims:self.intercept+=value
            elif len(dims)==1:self.main[dims[0]][slices[0]]+=value
            else:
                if dims not in self.pair:self.pair[dims]=np.zeros(tuple(len(self.cuts[j])+1 for j in dims))
                self.pair[dims][tuple(slices)]+=value
        self.reference_bins=self.bin(X)
        self.marginal=[np.bincount(self.reference_bins[:,j],minlength=len(c)+1)/len(X)
                       for j,c in enumerate(self.cuts)]
        self.measure='unpurified';self.audit={}
        self.raw_prediction_error=float(np.max(abs(self.predict(X)-model.predict(X,output_margin=True))))
        if self.raw_prediction_error>(1e-5 if self.output_scale=='logit' else 3e-6):
            raise AssertionError(f'Extraction prediction mismatch {self.raw_prediction_error}')

    def bin(self,X):
        X=np.asarray(X,dtype=np.float32)
        if X.ndim!=2 or X.shape[1]!=self.n_features or not np.isfinite(X).all():
            raise ValueError('Finite array with the fitted feature count required')
        return np.column_stack([np.searchsorted(c,X[:,j],side='right') for j,c in enumerate(self.cuts)])

    def components(self,X):
        bins=self.bin(X)
        arrays=[v[bins[:,j]] for j,v in enumerate(self.main)]
        keys=[(j,) for j in range(self.n_features)]
        for (j,k),v in sorted(self.pair.items()):
            arrays.append(v[bins[:,j],bins[:,k]]);keys.append((j,k))
        return keys,np.column_stack(arrays)

    def predict(self,X):
        return self.intercept+self.components(X)[1].sum(1)

    def purified(self,measure='empirical_joint'):
        if measure not in ('empirical_joint','product_marginals'):raise ValueError(measure)
        out=copy.deepcopy(self);out.measure=measure;out.audit={}
        for (j,k),v in out.pair.items():
            if measure=='product_marginals': w=np.outer(out.marginal[j],out.marginal[k])
            else:
                idx=out.reference_bins[:,j]*v.shape[1]+out.reference_bins[:,k]
                w=np.bincount(idx,minlength=v.size).reshape(v.shape).astype(float)
            v,a,b,diag=purify_pair(v,w)
            out.pair[j,k]=v;out.main[j]+=a;out.main[k]+=b
            diag['occupied_cells']=int(sum(w.ravel()>0));diag['total_cells']=v.size
            out.audit[f'{j},{k}']=diag
        for j,v in enumerate(out.main):
            avg=float(v @ out.marginal[j]);out.main[j]-=avg;out.intercept+=avg
        return out

    def importance(self,X):
        keys,values=self.components(X)
        cov=np.cov(values,rowvar=False,ddof=0)
        variance=np.diag(cov); total=float(values.sum(1).var())
        summary=dict(empirical_prediction_variance=total,component_variance_sum=float(variance.sum()),
            cross_covariance_total=float(cov.sum()-variance.sum()),
            reconstruction_variance_error=float(abs(total-cov.sum())),
            max_main_mean=float(max(abs(values[:,:self.n_features].mean(0)))))
        return keys,variance,cov,summary
