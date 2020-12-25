#Scikit-learn doesn’t accept non-numerical features
#Scikit-learn: OneHotEncode()
#Pandas: get_dummies()
#Ex: df = pd.get_dummies(df)


#Exploring categorical features
# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()



#Creating dummy variables
# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region, drop the unneeded dummy variable
df_region = pd.get_dummies(df, drop_first = True)

# Print the new columns of df_region
print(df_region.columns)



#Regression with categorical features
# Import necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)



#Imputing missing data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = ’NaN’, strategy = ‘mean’, axis = 0)
imp.fit(X)
X = imp.transform(X)



#Imputing within a pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
imp = imputer(missing_values = ’NaN’, strategy = ‘mean’, axis = 0)
logreg = LogisticRegression()
steps = [(‘imputation’, imp),
		(‘logistic_regression’, logreg)]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
pipeline.fit(X_train, y_train) # fit on training set
y_pred = pipeline.predict(X_test) # predict on testing set
pipeline.score(X_test, y_test) # compute accuracy 


#ex:
# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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

                     precision    recall  f1-score   support

               0       0.99        0.96     0.98        85
               1       0.94        0.98     0.96        46

        avg / total    0.97        0.97     0.97        131



        
#Centering and Scaling (Normalizing)
Why scale your data?
* Many models use some form of distance to inform them
* Features on larger scales can unduly influence the model
    * ex: K-NN uses distance explicitly when making predictions
* We want features to be on the same scale
* Scaling improves model performances
* In binary features, scaling will have minimal impact

Ways to normalize your data
* Standardization: (X-mean)/variance
    * All features are centered around 0 and have variance 1
* Normalization: (X-min)/(max-min)
    * Minimum 0 and maximum 1
* Can also normalize so that data ranges from -1 to +1


#Centering and scaling in a pipeline (Compare Accuracy before & after scaling)
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(pipeline.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))
#Accuracy with Scaling: 0.7700680272108843
#Accuracy without Scaling: 0.6979591836734694



#Bringing it all together I: Pipeline for classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.SVM import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

# Setup the pipeline
steps = [('scaler', StandardScaler()),
		('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyper-parameter space
# C controls the regularization strength, gamma controls the kernel coefficient
parameters = {'SVM__C':[1, 10, 100],
             		   'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

Output:
    Accuracy: 0.7795918367346939
                 precision   recall  f1-score   support
    
        False      0.83      0.85      0.84       662
        True       0.67      0.63      0.65       318
 avg / total       0.78      0.78      0.78       980
    
    Tuned Model Parameters: {'SVM__C': 10, 'SVM__gamma': 0.1}



#Bringing it all together II: Pipeline for regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         	('scaler', StandardScaler()),
         	('elasticnet', ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyper-parameter space
parameters = {'elasticnet__l1_ratio': np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))

#Output:
#    Tuned ElasticNet Alpha: {'elasticnet__l1_ratio': 1.0}
#    Tuned ElasticNet R squared: 0.8862016570888217
