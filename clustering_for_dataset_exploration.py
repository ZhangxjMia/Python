#Unsupervised Learning
#* Unsupervised learning finds patterns in data
#    * clustering customers by their purchases (clustering)
#    * compressing the data using purchase patterns (dimension reduction)

#Supervised Learning vs Unsupervised Learning
#* Supervised learning finds patterns for a prediction task
#    * classify tumors as benign or cancerous (labels)
#* Unsupervised learning finds patterns in data but without a specific prediction task in mind (pure pattern discovery)



#Clustering 2D points
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters = 3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)

[1 2 0 1 2 1 2 2 2 0 1 2 2 0 0 2 0 0 2 2 0 2 1 2 1 0 2 0 0 1 1 2 2 2 0 1 2
     2 1 2 0 1 1 0 1 2 0 0 2 2 2 2 0 0 1 1 0 0 0 1 1 2 2 2 1 2 0 2 1 0 1 1 1 2
     1 0 0 1 2 0 1 0 1 2 0 2 0 1 2 2 2 1 2 2 1 0 0 0 0 1 2 1 0 0 1 1 2 1 0 0 1
     0 0 0 2 2 2 2 0 0 2 1 2 0 2 1 0 2 0 0 2 0 2 0 1 2 1 1 2 0 1 2 1 1 0 2 2 1
     0 1 0 2 1 0 0 1 0 2 2 0 2 0 0 2 2 1 2 2 0 1 0 1 1 2 1 2 2 1 1 0 1 1 1 0 2
     2 1 0 1 0 0 2 2 2 1 2 2 2 0 0 1 2 1 1 1 0 2 2 2 2 2 2 0 0 2 0 0 0 0 2 0 0
     2 2 1 0 1 1 0 1 0 1 0 2 2 0 2 2 2 0 1 1 0 2 2 0 2 0 0 2 0 0 1 0 1 1 1 2 0
     0 0 1 2 1 0 1 0 0 2 1 1 1 0 2 2 2 1 2 0 0 2 1 1 0 1 1 0 1 2 1 0 0 0 0 2 0
     0 2 2 1]



#Inspect your clustering
# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c = labels, alpha = 0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker = 'D', s = 50)
plt.show()

￼

#Evaluating a clustering

#Measuring clustering quality
#* Using only samples and their cluster labels
#* A good clustering has tight clusters
#* Samples in each cluster bunched together

#Inertia measures clustering quality
#* Measures how spread out the clusters are (lower is better)
#* Measures how far samples are from their centroids
#* After fit(), available as attribute inertia_
    * from sklearn.cluster import KMeans
    * model = KMeans(n_clusters = 3)
    * model.fit(samples)
    * print(model.inertia_)



#How many clusters of grain?
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

￼

#Evaluating the grain clustering
import pandas as pd
from sklearn.cluster import KMeans

# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
# Using .fit_predict() is the same as using .fit() followed by .predict().
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

#Output:
#varieties  Canadian  Kama  Rosa (wheat)
#labels                                           
#0             0        1          60
#1            68        9           0
#2             2       60          10



#Transforming features for better clusterings

#Piedmont wines dataset
#* 178 samples from 3 distinct varieties if red wine: Barolo, Grignolino, and Barbera
#* Features measure chemical composition e.g. alcohol content
#* Visual properties like ‘color intensity’

#Clustering the wines:
from sklearn.cluster import KMeans
model = KMeans(n_clusters = 3)
labels = model.fit_predict(samples)

#Clusters vs. varieties
df = pd.DataFrame({‘labels’ : labels,
				     ‘varieties’ : varieties})
ct = pd.crosstab(df[‘labels’], df[‘varieties’])

#Output:
#varieties		Barbera 	Barolo		Grignolino
#labels
#0			29		13		20
#1			0		46		1
#2			19		0		50

#The KMeans clusters don’t correspond well with the wine varieties
#The problem is:
#* The wine features have very different variances
#* Variance of a feature measures spread of its values	

#StandardScaler
#* In KMeans: feature variance = feature influence
#* To give every feature a chance, the data needs to be transformed so that features have equal variance
#* StandardScaler transforms each feature to have mean 0 and variance 1
#* Features are said to be ‘standardized’
￼

#sklearn StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy = True, with_mean = True, with_std = True)
samples_scaled = scaler.transform(samples)



#Pipeline combine multiple steps
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters = 3)
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)
labels = pipeline.predict(samples)

#with feature standardization:
#varieties		Barbera 	Barolo		Grignolino
#labels
#0			0		59			3
#1			48		0			3
#2			0		0			65



#Scaling fish data for clustering
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters = 4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, kmeans)



#Clustering the fish data
# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct
print(ct)

#Outout:
#    species  Bream  Pike  Roach  Smelt
#    labels                            
#    0          0     0      0      13
#    1          33    0      1       0
#    2          0     17     0       0
#    3          1     0     19       1



#Clustering stocks using KMeans
#While StandardScaler() standardizes features (such as the features of the fish data from the previous exercise) by removing the mean and scaling to unit variance, 
#Normalizer() rescales each sample - here, each company's stock price - independently of the other.

# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters = 10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)
