from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from tesimodules import Classificator as cs

class MyGridSearchCV:

    def __init__(self, model, c=[1, 10, 10000], sigmas=[.1, .25, .5, 5]):
        kf = KFold(n_splits=3, shuffle=True)

        self.clf = GridSearchCV(model, {'c':c, 'sigma': sigmas}, n_jobs=-1,cv=kf, verbose=10)

    def fit(self, values, labels, allLabels):
        self.clf.fit(values, labels, mu=cs.Get_mu(allLabels), adjustment=1.8e+03) 
        return self.clf 

    