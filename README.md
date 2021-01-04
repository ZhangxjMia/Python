# Python
> This repository contains all my notes and mini projects/cases written in Python.

## Unsupervised Learning
> **Unsupervised Learning in Python** on DataCamp.
* Visualization with hierarchical clustering and t-SNE
* Clustering for dataset exploration

### Code Example
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
