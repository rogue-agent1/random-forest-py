#!/usr/bin/env python3
"""Random forest classifier — ensemble of decision stumps."""
import random, math, sys
from collections import Counter

class Stump:
    def __init__(self): self.feat=0; self.thresh=0; self.left=0; self.right=0
    def fit(self, X, y, features):
        best = float('inf')
        for f in features:
            vals = sorted(set(row[f] for row in X))
            for i in range(len(vals)-1):
                t = (vals[i]+vals[i+1])/2
                l = [yi for xi,yi in zip(X,y) if xi[f]<=t]
                r = [yi for xi,yi in zip(X,y) if xi[f]>t]
                if not l or not r: continue
                gini = len(l)/len(y)*(1-sum((v/len(l))**2 for v in Counter(l).values())) + \
                       len(r)/len(y)*(1-sum((v/len(r))**2 for v in Counter(r).values()))
                if gini < best:
                    best=gini; self.feat=f; self.thresh=t
                    self.left=Counter(l).most_common(1)[0][0]
                    self.right=Counter(r).most_common(1)[0][0]
    def predict(self, x): return self.left if x[self.feat]<=self.thresh else self.right

class RandomForest:
    def __init__(self, n_trees=10, max_features=None):
        self.n_trees=n_trees; self.max_features=max_features; self.trees=[]
    def fit(self, X, y):
        n = len(X); d = len(X[0])
        mf = self.max_features or max(1, int(math.sqrt(d)))
        for _ in range(self.n_trees):
            idx = [random.randint(0,n-1) for _ in range(n)]
            bX = [X[i] for i in idx]; by = [y[i] for i in idx]
            feats = random.sample(range(d), min(mf, d))
            stump = Stump(); stump.fit(bX, by, feats); self.trees.append(stump)
    def predict(self, X):
        return [Counter(t.predict(x) for t in self.trees).most_common(1)[0][0] for x in X]

if __name__ == "__main__":
    random.seed(42)
    X = [[random.gauss(c*3,1), random.gauss(c*3,1)] for c in range(3) for _ in range(20)]
    y = [c for c in range(3) for _ in range(20)]
    rf = RandomForest(n_trees=20); rf.fit(X, y)
    preds = rf.predict(X)
    acc = sum(p==t for p,t in zip(preds,y))/len(y)
    print(f"Accuracy: {acc:.1%}")
