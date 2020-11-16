import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.model_selection import *
from sklearn.decomposition import PCA
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

df = pd.read_csv('data.csv', header = 0)
df = df.drop(['id'], axis=1)
diagnosis = df['diagnosis']
df['catNB'] = pd.factorize(df['diagnosis'].values)[0]
dfLabels = df['catNB'] 
labels = ['M', "B"]

# print(df.head())

features = ['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean'] # Features I will use 
# 'perimeter_mean', 'area_mean',

bestAccuracy = 0.0
bestParams = None
bestCombination = []
bestDF = None
bestModel = None

def rSubset(arr, r): 
    # return list of all subsets of length r 
    # to deal with duplicate subsets use  
    # set(list(combinations(arr, r))) 
    return list(combinations(arr, r)) 

for nNumberOfFeatures in range(2, len(features)+1):
    possibleCombinations = rSubset(features, nNumberOfFeatures) # Creates complete list of all combinations of
    for combination in possibleCombinations:
        currentDF = pd.DataFrame()
        i = 0
        for feature in combination:
            currentDF.insert(i, feature, df[feature].to_list(), True)
            i += 1

        x = currentDF
        y = dfLabels

        k = 10 # Number of splits
        maxRange = 10
        model = DecisionTreeClassifier(random_state = 0)
        # Set up the grid of parameters
        hyperParams = {'criterion': ['gini', 'entropy'],
                    'max_depth': range(1, maxRange)}
        # Train the model in the grid search
        kFolds = StratifiedKFold(n_splits = k, shuffle = True, random_state = 0)
        search = GridSearchCV(model, hyperParams, cv = kFolds, scoring = 'accuracy')
        search = search.fit(x, y)
        # Show some of the results using a dataframe
        dfResults = pd.DataFrame(search.cv_results_)
        results = dfResults[['param_criterion', 'param_max_depth', \
                    'mean_test_score', 'std_test_score']]
        model = search.best_estimator_
        # print(search.best_params_)

        maxDepthRange = range(1, maxRange)
        trainScores, testScores = validation_curve(model, x, y, \
                    param_name = 'max_depth', param_range = maxDepthRange, \
                    cv = kFolds, scoring = 'accuracy')
        trainScoresMean = np.mean(trainScores, axis=1)
        trainScoresStd = np.std(trainScores, axis=1)
        testScoresMean = np.mean(testScores, axis=1)
        testScoresStd = np.std(testScores, axis=1)

        filteredTestScoresMean = [] # Include all test scores that are within STD of train scores mean
        filteredTestScoresMeanIndex = [] 
        for i in range(0, len(trainScoresMean)):
            if testScoresMean[i] + testScoresStd[i] >= trainScoresMean[i]:
                filteredTestScoresMean.append(testScoresMean[i])
                filteredTestScoresMeanIndex.append(i)

        print("Test Scores Left:", len(filteredTestScoresMean))

        bestScore = 0.0
        bestScoreIndex = 0
        for i in range(0, len(filteredTestScoresMean)):
            if filteredTestScoresMean[i] > bestScore:
                bestScore = filteredTestScoresMean[i]
                bestScoreIndex = filteredTestScoresMeanIndex[i]

        if len(filteredTestScoresMean) == 0:
            bestScore = testScoresMean[search.best_params_['max_depth'] -1 ]
            bestScoreIndex = search.best_params_['max_depth'] - 1
        
        highestAccuracy = (bestScore, bestScoreIndex) # Percent, depth 
        # Add standard deviation checking

        search.best_params_['max_depth'] = bestScoreIndex + 1

        print("Accuracy with", combination, ":", highestAccuracy[0])
        print("Params:", search.best_params_, '\n')

        if highestAccuracy[0] > bestAccuracy:
            bestAccuracy = highestAccuracy[0]
            bestCombination = combination
            bestDF = currentDF
            bestParams = search.best_params_
            bestModel = DecisionTreeClassifier(criterion=search.best_params_['criterion'], max_depth=search.best_params_['max_depth'])

            confusion = np.zeros((2,2))
            scores = cross_val_score(bestModel, x, y, cv = kFolds)
            predicted = cross_val_predict(bestModel, x, y, cv = kFolds)
            accuracy = scores.mean()
            print(accuracy)
            confusion = confusion_matrix(y, predicted)
            confDF = pd.DataFrame(confusion)
            confDF.columns = labels
            confDF.index = labels

            fig, ax = plt.subplots() # figsize=(6,10)
            plt.title('Confusion matrix with best accuracy using Decision Tree')
            ax = sns.heatmap(confDF, annot = True, square=True, linewidth=3)
            fig.savefig('Best Heatmap Stratified K-Fold with Decision Tree.png')
            plt.close()

        # print("Accuracy:", testScoresMean)

print("Best Accuracy:", bestAccuracy)
print("Feature Combination of Best Accuracy:", bestCombination)
print("Best Parameters of Best Accuracy:", bestParams)

x = bestDF
y = dfLabels
model = bestModel

maxDepthRange = range(1, maxRange)
trainScores, testScores = validation_curve(model, x, y, \
            param_name = 'max_depth', param_range = maxDepthRange, \
            cv = kFolds, scoring = 'accuracy')
trainScoresMean = np.mean(trainScores, axis=1)
trainScoresStd = np.std(trainScores, axis=1)
testScoresMean = np.mean(testScores, axis=1)
testScoresStd = np.std(testScores, axis=1)

print(testScoresMean)

plt.plot(maxDepthRange, trainScoresMean, label = 'Training Score', \
         color = 'darkorange', lw = 1, marker = 'o', markersize = 3)
plt.fill_between(maxDepthRange, trainScoresMean - trainScoresStd, \
                 trainScoresMean + trainScoresStd, alpha = 0.2, \
                 color = 'darkorange', lw = 1)
plt.plot(maxDepthRange, testScoresMean, label = 'Validation Score', \
         color = 'navy', lw = 1, marker = 's', markersize = 3)
plt.fill_between(maxDepthRange, testScoresMean - testScoresStd, \
                 testScoresMean + testScoresStd, alpha = 0.2, \
                 color = 'navy', lw = 1)


plt.title("Validation Curve with Best Decision Tree Classifier")
plt.xlabel('Max Depth Parameter')
plt.ylabel("Score")
plt.legend(loc="best")

plt.show()

scores = cross_val_score(model, x, y, cv = kFolds)
predicted = cross_val_predict(model, x, y, cv = kFolds)
accuracy = scores.mean()
confusion = confusion_matrix(y, predicted)
confDF = pd.DataFrame(confusion)
confDF.columns = labels
confDF.index = labels

fig, ax = plt.subplots() # figsize=(6,10)
plt.title('Confusion Matrix with Best Accuracy (Decision Tree Classifier)')
ax = sns.heatmap(confDF, annot = True, square=True, linewidth=3)
fig.savefig('Best Heatmap Decision Tree.png')
plt.close()

'''plt.figure()
model.fit(x, y)
plot_tree(model, filled=True)
plt.show()'''

#PCA Analysis
pca = PCA(n_components = 2)
print("bestDF:", bestDF)
x = bestDF.values
print(x)
pca.fit(x)
print('Components using Scikit-learn =')
print(pca.singular_values_)
print(pca.components_)
xP = pca.transform(x)
currentDFPCA = pd.DataFrame(data = xP, columns = ['eig1', 'eig2'])

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
print(model)
model.fit(bestDF,dfLabels)
Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])) # Predicting after inversing the PCA
Z = Z.reshape(xx.shape)

# Defining the colors to use
colors=['blue', 'red']
markers = ['s', 'p']
cmap = ListedColormap(colors)

currentDFPCA['diagnosis'] = diagnosis

# Plotting
#print(currentDFPCA.head())
ax = sns.scatterplot(data=currentDFPCA, x='eig1', y='eig2', hue = "diagnosis", palette=colors, style="diagnosis", markers=markers)
ax.contourf(xx, yy, Z, alpha=0.2, cmap=cmap) # Paint between boarders
ax.set_title('Best Decision Tree Decision Regions plotted on PCA Space')
plt.show()