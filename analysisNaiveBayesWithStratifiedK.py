import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import *
from sklearn.naive_bayes import *
from sklearn.model_selection import *
from sklearn.decomposition import PCA
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

df = pd.read_csv('data.csv', header = 0)
df = df.drop(['id'], axis=1)
diagnosis = df['diagnosis']
labels = ['M', "B"]

features = ['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean'] # Features I will use 
#df = df.drop(['diagnosis'], axis=1)
#features = df.columns

def rSubset(arr, r): 
    # return list of all subsets of length r 
    # to deal with duplicate subsets use  
    # set(list(combinations(arr, r))) 
    return list(combinations(arr, r)) 

# Function to convert   
def listToString(s):  
    # initialize an empty string 
    str1 = ""  
    # traverse in the string   
    for ele in s: 
        str1 += ele + ' '   
    # return string   
    return str1
        

bestAccuracy = 0.0
bestCombination = []
bestDF = None


for nNumberOfFeatures in range(2, len(features)+1):
    possibleCombinations = rSubset(features, nNumberOfFeatures) # Creates complete list of all combinations of
    for combination in possibleCombinations:
        currentDF = pd.DataFrame()
        i = 0
        for feature in combination:
            currentDF.insert(i, feature, df[feature].to_list(), True)
            i += 1

        #print(currentDF.head())
        '''model = GaussianNB()
        model.fit(currentDF, diagnosis)
        accuracy = model.score(currentDF, diagnosis)
        predicted = model.predict(currentDF)
        confusion = confusion_matrix(diagnosis, predicted)
        confDF = pd.DataFrame(confusion)
        print(confDF.head())
        confDF.columns = labels
        confDF.index = labels'''
        x = currentDF
        y = diagnosis

        confusion = np.zeros((2,2))
        k = 10
        model = GaussianNB()
        kFolds = StratifiedKFold(n_splits = k, shuffle = True, random_state = 0)
        scores = cross_val_score(model, x, y, cv = kFolds)
        predicted = cross_val_predict(model, x, y, cv = kFolds)
        accuracy = scores.mean()
        confusion = confusion_matrix(y, predicted)
        confDF = pd.DataFrame(confusion)
        confDF.columns = labels
        confDF.index = labels

        print("Accuracy:",accuracy)

        print("Accuracy with", combination, ":", accuracy)
        if accuracy > bestAccuracy:
            bestAccuracy = accuracy
            bestCombination = combination
            bestDF = currentDF
            fig, ax = plt.subplots() # figsize=(6,10)
            plt.title('Confusion matrix with Best Accuracy (Naive Bayes Classifier)')
            ax = sns.heatmap(confDF, annot = True, square=True, linewidth=3)
            fig.savefig('Best Heatmap using NBC.png')
            plt.close()



print("Best Accuracy:", bestAccuracy) # 0.9226713532513181
print("Combination:", bestCombination) # ('radius_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean')

#PCA Analysis
pca = PCA(n_components = 2)
x = bestDF.values
pca.fit(x)
print('Components using Scikit-learn =')
print(pca.singular_values_)
print(pca.components_) 
xP = pca.transform(x)
currentDFPCA = pd.DataFrame(data = xP, columns = ['eig1', 'eig2'])

#CB factorize so labels are string --> integers
df['catNB'] = pd.factorize(df['diagnosis'].values)[0]
dfLabels = df['catNB'] 

#print(dfLabels)
X = currentDFPCA
Y = dfLabels # Labels

# Plotting decision regions
X = X.to_numpy()
#print("X:", X)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 #Compute boundaries of painting space
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/1000),
                    np.arange(y_min, y_max, (y_max-y_min)/1000)) #tesselation 0.1 - resolution

f, ax = plt.subplots(figsize=(10, 8))
print(np.c_[xx.ravel(), yy.ravel()].shape)
model.fit(bestDF, dfLabels)
Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])) # Predicting after inversing the PCA
Z = Z.reshape(xx.shape)

# Defining the colors to use
colors=['blue', 'red']
markers = ['s', 'p']
cmap = ListedColormap(colors)

currentDFPCA['diagnosis'] = df['diagnosis']

# Plotting
#print(currentDFPCA.head())
ax = sns.scatterplot(data=currentDFPCA, x='eig1', y='eig2', hue = "diagnosis", palette=colors, style="diagnosis", markers=markers)
ax.contourf(xx, yy, Z, alpha=0.2, cmap=cmap) # Paint between boarders
ax.set_title('Best NBC Decision Regions plotted on PCA Space')
f.savefig("Best NBC Decision Regions plotted on PCA Space.png")
plt.show()