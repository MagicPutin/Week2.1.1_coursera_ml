import sklearn.neighbors
import sklearn.preprocessing
from scipy.linalg.tests.test_fblas import accuracy
import pytest
from sklearn import svm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

row_data = open('data/wine.data', 'r')
row_data = row_data.read()

# preparing data
data = [[i for i in j.split(',')] for j in row_data.split('\n')]
feature = [i[1:] for i in data]
wine_class = [i[0] for i in data]

# cross validation of data
# generator of folding
kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
clf = svm.SVC(kernel='linear', C=1)
# yeah, thats works, but I need to build it more automatically
# so, yesterday, I'll write smth to automate this
for k in range(1,51):
    k_neighbors = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    print(k, round(sklearn.model_selection.cross_val_score(estimator=k_neighbors, X=feature, y=wine_class, cv=kf).mean(), 2))

