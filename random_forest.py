#!/usr/bin/env python3
"""Random forest — bagging + feature subsampling."""
import random
from collections import Counter

class Node:
    def __init__(self, f=None, t=None, l=None, r=None, label=None):
        self.f=f;self.t=t;self.l=l;self.r=r;self.label=label

def gini(y):
    n=len(y);c=Counter(y); return 1-sum((v/n)**2 for v in c.values())

def build(X, y, max_depth=10, max_features=None):
    if len(set(y))==1: return Node(label=y[0])
    if max_depth==0 or len(y)<2: return Node(label=Counter(y).most_common(1)[0][0])
    d=len(X[0]); feats=random.sample(range(d),min(max_features or int(d**0.5+1),d))
    best_g,bf,bt=-1,None,None; n=len(y); pg=gini(y)
    for f in feats:
        vals=sorted(set(X[i][f] for i in range(n)))
        for i in range(len(vals)-1):
            t=(vals[i]+vals[i+1])/2
            ly=[y[j] for j in range(n) if X[j][f]<=t]; ry=[y[j] for j in range(n) if X[j][f]>t]
            if not ly or not ry: continue
            g=pg-len(ly)/n*gini(ly)-len(ry)/n*gini(ry)
            if g>best_g: best_g=g;bf=f;bt=t
    if bf is None: return Node(label=Counter(y).most_common(1)[0][0])
    li=[i for i in range(n) if X[i][bf]<=bt]; ri=[i for i in range(n) if X[i][bf]>bt]
    return Node(bf,bt,build([X[i] for i in li],[y[i] for i in li],max_depth-1,max_features),
                      build([X[i] for i in ri],[y[i] for i in ri],max_depth-1,max_features))

def pred1(node, x):
    if node.label is not None: return node.label
    return pred1(node.l if x[node.f]<=node.t else node.r, x)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10): self.n=n_trees; self.md=max_depth; self.trees=[]
    def fit(self, X, y):
        n=len(X)
        for _ in range(self.n):
            idx=[random.randint(0,n-1) for _ in range(n)]
            self.trees.append(build([X[i] for i in idx],[y[i] for i in idx],self.md))
    def predict(self, x):
        votes=Counter(pred1(t,x) for t in self.trees)
        return votes.most_common(1)[0][0]

def main():
    random.seed(42)
    X=[[i,i] for i in range(20)]; y=[0]*10+[1]*10
    rf=RandomForest(10); rf.fit(X,y)
    print(f"Predict [5,5]:{rf.predict([5,5])}, [15,15]:{rf.predict([15,15])}")

if __name__=="__main__":main()
