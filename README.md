# News Categorization

**Background**:
From the beginning, since first news paper printed, every news that makes into page has had a specific section allotted to it. The newspaper style, news sections, format etc.. have been changed over the time but not the categorization of the news and it still carried over even in to the digital version of newspaper. News articles are not limited to few topics, it covers a wide range of interest from politcs to sports to moveis and so on. For long time, this process of categorization news was done manually by people and used to allot news to respective section(category). With digitalization of news paper, the news gets updated every moment and allocating to them to appropriate category can be cumbersome task.

Text categorization or classification is a way of assigning documents to one or more predefined categories. This helps news editors, end users to allot and find the appropriate news category.
The main problem we are going to handle as part of this project is, to categorize news based on headlines and short description by using supervised machine learning classifiers.

**How to Solve this problem** - To avoid manual news categorization, with help of latest technology, Natural Language Processing and machine leanring, this problem will tackled to classify and predict which category a piece of news will fall into based on the news headline and short description.

**What model will be built for and How it would help** - In order to solve the manual news categorization problem, A machine learning model will be built using supervised machine learning techniques, that would learn from existing news headlines and short description and predict the news category appropriately. With the help of this model the news categorization can be automated and it would save manual work and help users to read the news of their interest in right section.

**Data Description**:
This dataset contains around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. These news headlines and short descriptions were posted in different newspapers in the past, corresponding news categories were also collected for each news which was published under that category.
Each news headline has a corresponding category. Categories and corresponding article have different counts. These categories are predefined the time of collecting the data and assigned them accordingly.

**Model Selection**:

When working on data science project to find insights from the data or appropriately categorize the data based on given inputs, we often need to use machine learning algorithms that helps resolve the problem. There are several machine learning algorithms that can be used but to select appropriate algorithms based on the data and problem, it is crucial step in the project and required information about the algorithms and how it works. 
Why model selection is important? In machine learning projects model selection is a process data scientist use to compare the relative value of different machine learning models and determine which one is the best fit for the observed data.  
Since, the problem statement of this project falls under classification, we are going to use several different supervised machine learning classifiers to train on news data set and evaluate each one of them for accuracy and several different evaluation metrics.
The first machine learning algorithm selected for classifying categories for news data set is Naive Bayes Classifier. 

1. **Naïve Bayes Classifier** - It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. It works on Bayes theorem of probability to predict the class of unknown data sets.
**Reason to use Naïve Bayes Classifier** -
- It is easy and fast to predict the class of the test data set. It also performs well in multi-class prediction. When assumption of independence holds, a Naive Bayes classifier performs better compared to other models like logistic regression.
- Because of the class independence assumption, naive Bayes classifiers can quickly learn to use high dimensional features with limited training data compared to more sophisticated methods.
- It is the most applied commonly to text classification. Though it is a simple algorithm, it performs well in many text classification problems.
- It performs well in case of categorical input variables compared to numerical variable(s). Because for numerical variable, normal distribution is a strong assumption.

Based on the advantages of Naïve Bayes Classifier, it helps to resolve our problem statement and since our training data is text (converted to vectors with high dimensions), this algorithm best suits for building model.

2. **Linear Support Vector Machine** – SVMs are a set of supervised learning methods used for classification, regression, and outliers’ detection.
Linear SVM is used for linearly separable data, which means if a dataset can be classified into two classes by using a single straight line, then such data is termed as linearly separable data, and classifier is used called as Linear SVM classifier. For multiclass classification, the same principle is utilized after breaking down the multiclassification problem into multiple binary classification problems. 

The advantages of support vector machines are:
- Effective in high dimensional spaces.
- Still effective in cases where number of dimensions is greater than the number of samples.
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

3. **Logistic Regression** - Logistic regression is a classification algorithm, used when the value of the target variable is categorical in nature. Logistic regression is most used when the data in question has binary output, so when it belongs to one class or another, or is either a 0 or 1. This algorithm can be used for multiclass classification because multiclass classification is built on the binary classification. The approach used for multi-class classification is one vs all method.

4. **XGBoost Classifier** - It is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.
It is more apt for multi-class classification task. By default, XGBClassifier or many Classifiers uses objective as binary but what it does internally is classifying (one vs rest).
**Advantages of XGBoost Classifier** – 
- It is Highly Flexible and supports regularization
- It uses the power of parallel processing and faster than Gradient Boosting
- It is designed to handle missing data with its in-build features.
- It Works well in small to medium dataset

**Model Evaluation**: 

Model Evaluation is an integral part of the model development process. It helps to find the best model that represents our data and how well the chosen model will work in the future. 
Model evaluation aims to estimate the generalization accuracy of a model on future (unseen/out-of-sample) data.
To evaluate the performance of the model chosen for classifying the news categories, we are going to use metrics such as – accuracy score, confusion matrix, precision, recall and f1 score etc.
Since the classes in our data sets are balanced, so accuracy is the most used evaluation metric for classification problems with balanced classes.

**Accuracy** – It is one of the common evaluation metrics in classification problems, that is the total number of correct predictions divided by the total number of predictions made for a dataset. Accuracy is useful when the target class is well balanced.

**Confusion Matrix** – A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes. The matrix compares the actual target values with those predicted by the machine learning model. It is a square matrix whose dimensions depend on the number of classes we have in our model. It is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.

**Why to use Confusion Matrix** - Classification accuracy alone can be misleading if we have an unequal number of observations in each class or if we have more than two classes in your dataset. Our data sets contain more than two classes, so Confusion Matrix is the right evaluation metrics.

**Precision** – Precision answers the question of “what proportion of predicted positives are truly positive?” The precision is calculated by dividing the true positives by the sum of true positives and false positives.

**Recall** – Recall answers the question of “what proportion of actual positives are correctly classified?” It is calculated by dividing the number of true positives by the sum of true positives and false negatives.

**F1 score** – Due to their nature, precision and recall are in a trade-off relationship. You may have to optimize one at the cost of the other. This is where the F1 score comes in. It is calculated by taking the harmonic mean of precision and recall and ranges from 0 to 1. F1 score is using harmonic mean because harmonic mean has a nice arithmetic property representing a truly balanced mean.
Also, we are going to plot confusion matrix for all classes to see the accuracy of classes with respect to other classes and precision, recall and f1 score are calculated together using Classification Report.


**Conclusion:**

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.
It is more apt for multi-class classification task. By default,XGBClassifier or many Classifier uses objective as binary but what it does internally is classifying (one vs rest).
