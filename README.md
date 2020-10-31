# Restaurant Inspection Predictions

### Predicting whether a restaurant in Boston will pass its health inspection, given its prior performane pattern on the same, coupled with its standing in Yelp.

### Motivation

A huge number of food illness outbreaks are sourced back to lack of hygienic practices within a restaurant and the manner in which it operates. To amerliorate this problem, strict regulations exist for F&B establishments when it comes to health standards. City officials carry out regular inspections to monitor the same; thus the work done by them is of huge importance. Unfortunately, like most things, there are never sufficient resources to be able to execute on number of inspections required. Hence, can we apply Data Science to help alleviate this problem?

In this project, I aim to predict restaurants that are most likely to Pass/Fail on a health inspection using classification machine learning models. The predicted outcome can help the authorities better prioritize resources towards restaurants that are expected to Fail. 

### Learning Goals & Tools Used

1. Data Processing, Modelling & Evaluation
    * Balanacing datasets (Random Over Sampler, SMOTE, ADASYN, Random Under Sampler)
    * Classification models (KNearest Neighbors, Logistic Regression, Gaussian Naive Bayes, Decision Trees, Random Forest Classifier, Gradient Boosted Classifier)
    * Evaluation Metrics (Accuracy, Precision, Recall, F1 score, ROC-AUC curves)
    * Hyperparameter tuning & model selection
2. Data Querying using PostgreSQL
3. Dynamic Visualizations
    * Tableau
    * Streamlit
  
A soft goal is to start creating \*.py files that store repeated function calls. These are intended to provide practice with writing code in a scalable manner that can also be re-used easily for future endeavors.

### Data Sources

* [Boston's Food Establishment Inspections Data](https://data.boston.gov/dataset/food-establishment-inspections) 
* Yelp Data (queried using their API)

### Approach

**Target Variable**

The results of a restaurant can be encoded using 10 different values. To reduce the prediction to a binary class, 2 result types will be mapped to a 'Pass' result, 4 to a 'Fail' result and the remaining to misc. aspects that will be dropped from the dataset. 

The latest inspection result for a given result will be used as the target variable.  


**Data Collection, Cleaning, EDA & Feature Engineering**

*Inspection Data*
Each restaurant gets inspected multiple times which has an overall result and within each inspection, a restaurant can have multiple violations. The historical inspections data contains a new line of entry for each violation within a given inspection for each restaurant. The overall result & result data of each inspection is recorded on every line, while the violation itself changes line to line.  

Some of the post important features here that will be relevant later can be read about [here]

The following steps were taken to wrangle the data:
1. Using the license number (of an establishment) & the result date (of an inspection) as unique IDs, all duplicate rows were dropped. Thus only one row per inspection for a given establishment was kept.  

2. The number of the days since the previous inspection was calculated. Inspections executed in less than 90 days were labeled as 'Re-inspections'. Restaurants get inspected no sooner than 90 days since the previous inspection. If they are inspected anytime sooner, it is due to them failing the prior inspection, that requires a follow-up.  

3. A majority of restaurants were found to 'Pass' their 'Re-inspection' but fail their 'Routine Inspection', as seen from the graph below.

Due to this stark distinction, only 'Routine Inspections' were kept for further analysis.  

The overall data was reduced to one row per restaurant and the following features were constructed:
1. Most recent prior inspection result
2. Second most recent prior inspection result
3. Total # of Inspections
3. # of Total Inspections 
4. # of Inspections Pass
5. Ratio of Inspections Passed
6. Ratio of Inspections Failed

*Yelp Data*
Using the Yelp API, a given restaurant's name & geographical information was used to get a potential match for the restaurant in Yelp's Database.  
The restuarant's current rating, # of reviews, price category & food category were obtained.

Finally, all categorical data was imputed & missing values were dealt with.

**Classification Modeling, Optimization Processing & Hyperparameter Tuning**

The following models from the sci-kit learn library were used to generate predictions - **KNearest Neighbors, Logistic Regression,** and **Random Forest Classifier.**
Based on initial results, features were dropped iteratively using Regularization, Feature Importances & Coefficients, while evaluating the F1 score, Precision & Recall.
Additonally, since the dataset was imbalanced (with a far greater number of fail predictions), oversampling via **RandomOverSampler** was used to create a balanced dataset.
Finally, GridSearchCV was deployed to find the most optimized parameters for the classification models, using an F1 scoring method.

**Scoring**

The **F1 score** was chosen as the evaluation metric to optimize. This is because it was vital to get as many predictions as possible correct, while minimizing the number of wrong predictions. Since, regardless of whether the restaurant Passes or Fails, it was important the prediction was done correctly in either instance. 

### Results





