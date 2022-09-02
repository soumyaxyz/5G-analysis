from mrmr import mrmr_classif
from sklearn.datasets import make_classification
import pandas as pd

# create some data


X = pd.DataFrame(X)
Y = pd.Series(Y)

# use mrmr classification
selected_features = mrmr_classif(X, Y, K = 10)

def getY(labels):
	Y = []
	for y in labels:
		Y.append(y[1]+(y[0]/-100))
	return np.array(Y)