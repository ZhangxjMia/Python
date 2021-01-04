# Python
> This repository contains all my notes and mini projects/cases written in Python.

## Unsupervised Learning
> **Unsupervised Learning in Python** on DataCamp.
* Visualization with hierarchical clustering and t-SNE
* Clustering for dataset exploration

#### Code Example
```Python
#Clustering the wines:
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
labels = model.fit_predict(samples)

#Clusters vs. varieties
df = pd.DataFrame({‘labels’ : labels, ‘varieties’ : varieties})
ct = pd.crosstab(df[‘labels’], df[‘varieties’])
```

## Supervised Learning
> **Supervised Learning with Scikit-Learn** on DataCamp
* Preprocessing and Pipelines
* Classification
* Regression
* Fine-tuning Model

#### Code Example
```Python
# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))
```
