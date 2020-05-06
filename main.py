import sklearn.neighbors
import sklearn.preprocessing

row_data = open('data/wine.data', 'r')
row_data = row_data.read()

# preparing data
data = [[i for i in j.split(',')] for j in row_data.split('\n')]
print(data)
feature = [i[1:] for i in data]
wine_class = [i[0] for i in data]
print(feature)
print(wine_class)

# cross validation of data
kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
print(sklearn.model_selection.cross_val_score(estimator=kf, X=feature, y=wine_class))
"""for k in range(1,51):
    sklearn.neighbors.KNeighborsClassifier(n_neighbors=k,)"""

