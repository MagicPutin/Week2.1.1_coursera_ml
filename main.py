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

# without scaling
max_score = 0
best_k = 0
for k in range(1,51):
    k_neighbors = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    score = sklearn.model_selection.cross_val_score(estimator=k_neighbors, X=feature, y=wine_class, cv=kf).mean()
    if score > max_score:
        max_score = score
        best_k = k

ans = open('answers/answer_1', 'w')
ans.write(str(best_k))

ans = open('answers/answer_2', 'w')
ans.write(str(round(max_score, 2)))

# with scaling
feature = sklearn.preprocessing.scale(feature)
max_score = 0
best_k = 0
for k in range(1,51):
    k_neighbors = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    score = sklearn.model_selection.cross_val_score(estimator=k_neighbors, X=feature, y=wine_class, cv=kf).mean()
    if score > max_score:
        max_score = score
        best_k = k
ans = open('answers/answer_3', 'w')
ans.write(str(best_k))

ans = open('answers/answer_4', 'w')
ans.write(str(round(max_score, 2)))
