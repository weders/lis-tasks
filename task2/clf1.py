import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn import neighbors, svm ,ensemble
import seaborn as sns
from scipy import stats
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score , cross_val_predict




#read train data in panda data frame
data_set = pd.read_csv("./train.csv" , index_col = 0  )
data_train = data_set[:850]
data_test = data_set[850:]
#extract features and labels
features_columns = list(data_set)[1:]
features_train = data_train[features_columns]
features_test = data_test[features_columns]
targets_train = data_train['y']
targets_test = data_test['y']
#read test_data --> final
data_final = pd.read_csv("./test.csv" , index_col = 0  )

#features selection 
#scatter_matrix(features_train)
#plt.show()

features_train=features_train.drop(['x8','x9','x14'], axis=1)
features_test=features_test.drop(['x8','x9','x14'], axis=1)
data_final=data_final.drop(['x8','x9','x14'], axis=1)
"""
correlations = features_train.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,15,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(features_columns)
ax.set_yticklabels(features_columns)
plt.show()

# visualizing data
# mean and std of the features
sns.set(color_codes=True)
for features in features_columns:
        sns.distplot(features_train[features])
        plt.show()

#boxplot 
sns.set_style("whitegrid")
ax = sns.boxplot(x=features_train["x2"])
plt.show()

#x vs y
sns.regplot(features_train['x2'],targets_train)
plt.show()
"""

#random forrest


clf_rndf= ensemble.RandomForestClassifier(n_estimators= 750,criterion ='entropy', min_samples_leaf=1)
clf_rndf = clf_rndf.fit(features_train,targets_train)
scores = cross_val_score(clf_rndf,features_train, targets_train, cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Score on test set: %0.2f" % clf_rndf.score(features_test,targets_test))




#print to csv
pred = clf_rndf.predict(data_final)
data_final['y']= pred
result = data_final['y']
result.to_csv('result.csv',header=['y'])
