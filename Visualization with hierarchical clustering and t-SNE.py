#two USL techniques for visualization: t-SNE & Hierarchical clustering


#Hierarchical clustering with SciPy
#Given samples (the array of scores) and country_names
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method = ‘complete’)
dendrogram(mergings,
			labels = country_names,
			leaf_rotation = 90,
			leaf_font_size = 6)
plt.show()



#Hierarchical clustering of the grain data
# Perform the necessary imports
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(samples, method = 'complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()




#Hierarchies of stocks
# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method = 'complete')

# Plot the dendrogram
dendrogram(mergings,
           labels = companies,
           leaf_rotation = 90,
           leaf_font_size = 6)
plt.show()




Dendrograms show cluster distances
	•	Height on dendrogram = distance between merging clusters
	•	E.g. clusters with only Cyprus and Greece had distance approx. 6
	•	This new cluster distance approx. 12 from cluster with only Bulgaria




Distance between clusters
	•	Defined by a “linkage method”
	•	In “complete” linkage: distance between clusters is max. distance between their samples
	•	Specified via method parameter, e.g. linkage(samples, method = ‘complete’)



#Different linkage, different hierarchical clustering!
# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage: mergings
mergings = linkage(samples, method = 'single')

# Plot the dendrogram
dendrogram(mergings, 
			labels = country_names, 
			leaf_rotation = 90, 
			leaf_font_size = 6)
plt.show()




Extracting cluster labels using fcluster
	•	Use the fcluster() function
	•	Returns a NumPy array of cluster labels

    
from scipy.cluster.hierarchy import linkage
mergings = linkage(samples, method = ‘complete’)
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 15, criterion = ‘distance’)
print(labels)

# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6, criterion = 'distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

Output:
varieties  Canadian wheat  Kama wheat  Rosa wheat
labels                                           
1   14  3   0
2   0   0   14
3   0   11  0


#Aligning cluster labels with country names
#Given a list if strings country_names:
import pandas as pd
pairs = pd.DataFrame({‘labels’: labels, ‘countries’: country_names})
print(pairs.sort_values(‘labels’))



t-SNE for 2-dimensional maps
	•	t-SNE = ’t-distributed stochastic neighbor embedding’
	•	Maps samples to 2D space (or 3D)
	•	Map approximately preserves nearness of samples
	•	Great for inspecting datasets

#t-SNE in sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate = 100)
transformed = model.fit_transform(samples)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c = species)
plt.show()

t-SNE has only fit_transform()
	•	Has a fit_transform() method
	•	Simultaneously fits the model and transforms the data
	•	Has no separate fit() or transform() methods
	•	Can’t extend the map to include new data samples
	•	Must start over each time

t-SNE learning rate
	•	Choose learning rate for the dataset
	•	Wrong choice: points bunch together
	•	It’s enough to try values between 50 and 200

Different every time
	•	t-SNE features are different every time
	•	Piedmont wines, 3 runs, 3 different scatter plots
	•	…however: The wine varieties (=colors) have same position relative to one another




#t-SNE visualization of grain dataset
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate = 200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c = variety_numbers)
plt.show()




#A t-SNE map of the stock market
# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate = 50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs, ys, alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()
