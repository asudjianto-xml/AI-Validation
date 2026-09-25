import json, numpy as np
from scipy.stats import spearmanr
from aiv.predictive import PredictiveSystem
from aiv.wsearch import *
from aiv.wsearch.gp import CompositeGP, interval_coverage
from aiv.wsearch.lossmodel import AuxLossModel, region_embeddings
from aiv.wsearch.constructor import Candidate
sysm=PredictiveSystem().fit(); feats=list(sysm.features)
disc,train=sysm.splits["discovery"],sysm.splits["train"]
r=sysm.row_loss(disc); std=Standardizer.fit(train,feats); Xs=std.transform(disc); Xr=disc[feats].to_numpy(float)
aux=AuxLossModel(depth=3,n_estimators=200).fit(Xr,r); Phi=aux.phi(Xr)
con=FixedSupportConstructor(Xs,m=171,metric="box"); ev=WeaknessEvaluator(r)
regs=con.build_batch(sample_candidates(Xs,2000,np.random.default_rng(0),"box"))
M=np.array([ev.contrast(g.member_idx) for g in regs]); base=regs[int(np.argmax(M))].candidate
LAMS=[0.0,0.1,0.25,0.5,0.75,0.9,1.0]
arch={"local_0.18":lambda g:[Candidate(base.z+0.18*g.normal(size=10),base.w) for _ in range(250)],
      "local_0.60":lambda g:[Candidate(base.z+0.60*g.normal(size=10),base.w) for _ in range(250)],
      "global_box":lambda g: sample_candidates(Xs,250,g,"box"),
      "global_cube":lambda g: sample_candidates(Xs,250,g,"cube")}
out={}
for tag,bld in arch.items():
    cands=bld(np.random.default_rng(11)); mem=[g.member_idx for g in con.build_batch(cands)]
    y=np.array([ev.contrast(i) for i in mem]); E=region_embeddings(Phi,mem)
    C=np.stack([c.as_tuple() for c in cands]); tr,te=np.arange(175),np.arange(175,250)
    rm,cv=[],[]
    for lam in LAMS:
        g=CompositeGP(lam=lam).fit(C[tr],E[tr],y[tr]); mu,sd=g.predictive(C[te],E[te])
        rm.append(float(np.sqrt(((mu-y[te])**2).mean()))); cv.append(interval_coverage(mu,sd,y[te],0.90))
    out[tag]={"lams":LAMS,"rmse":rm,"cov90":cv,"y_sd":float(y.std())}
    print(tag, [f"{v:.4f}" for v in rm], f"y_sd={y.std():.3f}", flush=True)
# tie counts and B1 pool ranks already measured elsewhere; record acquisition ties here
from aiv.wsearch.gp import expected_improvement
tr_c=sample_candidates(Xs,120,np.random.default_rng(0),"box")
tr_m=[g.member_idx for g in con.build_batch(tr_c)]
ytr=np.array([ev.contrast(i) for i in tr_m])
gp=CompositeGP(lam=0.0).fit(np.stack([c.as_tuple() for c in tr_c]),region_embeddings(Phi,tr_m),ytr)
best=tr_c[int(np.argmax(ytr))]
rng=np.random.default_rng(3)
pool=[Candidate(best.z+s*rng.normal(size=10),best.w) for s in np.repeat([0.02,0.05,0.15],400)]
pm=[g.member_idx for g in con.build_batch(pool)]
mu,sd=gp.predict(np.stack([c.as_tuple() for c in pool]),region_embeddings(Phi,pm))
ei=expected_improvement(mu,sd,ytr.max())
out["ties"]={"n_pool":len(pool),"distinct_members":len({tuple(a) for a in pm}),
             "distinct_ei":int(len(np.unique(np.round(ei,12)))),
             "within_1pct_of_max":int((ei>=0.99*ei.max()).sum())}
print(out["ties"], flush=True)
json.dump(out,open("runs/lambda_sweep.json","w"),indent=2)
