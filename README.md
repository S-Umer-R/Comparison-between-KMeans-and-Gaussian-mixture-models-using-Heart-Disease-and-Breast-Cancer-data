## Comparison Between Unsupervised Learning Algorithms

This repository contains Python code that demonstrates the application of K-means clustering and Gaussian Mixture Models (GMM) on the Heart Disease Cleveland UCI and Breast Cancer Wisconsin (Diagnostic) datasets. The objective of this project is to compare the performance of these clustering algorithms on tabular datasets and evaluate their accuracy using confusion matrices.

### Datasets Used
1. Heart Disease Cleveland UCI dataset:
   - This dataset contains various features related to heart disease conditions, including age, blood pressure, cholesterol level, etc.
   - The dataset is loaded using the pandas library.
   - Feature scaling is performed using MinMaxScaler to normalize the values between 0 and 1.
   - The dataset is split into features (X) and the target variable (y).

2. Breast Cancer Wisconsin (Diagnostic) dataset:
   - This dataset consists of features extracted from digitized images of breast mass.
   - The goal is to predict whether a tumor is malignant or benign.
   - The dataset is loaded using the pandas library.
   - Feature scaling is performed using MinMaxScaler to normalize the values between 0 and 1.
   - The dataset is split into features (X) and the target variable (y).

### K-means Clustering
1. Determining the Optimal Number of Clusters:
   - The elbow method is used to identify the optimal number of clusters (K).
   - The sum of squared errors (SSE) is calculated for different values of K ranging from 1 to 9.
   - A line plot is generated to visualize the SSE values against the number of clusters.

2. Clustering on Heart Disease Dataset:
   - The Heart Disease dataset is split into training and test sets using train_test_split.
   - K-means clustering is performed on the training set with a chosen number of clusters.
   - The resulting clusters are used to predict the labels for both the training and test datasets.
   - The accuracy scores are computed by comparing the predicted labels with the true labels.

3. Clustering on Breast Cancer Dataset:
   - The Breast Cancer dataset is split into training and test sets using train_test_split.
   - K-means clustering is performed on the training set with a chosen number of clusters.
   - The resulting clusters are used to predict the labels for both the training and test datasets.
   - The accuracy scores are computed by comparing the predicted labels with the true labels.

### Gaussian Mixture Models (GMM)
1. Clustering on Heart Disease Dataset:
   - The Heart Disease dataset is split into training and test sets using train_test_split.
   - GMM clustering is performed on the training set.
   - The resulting clusters are used to predict the labels for both the training and test datasets.
   - The accuracy scores are computed by comparing the predicted labels with the true labels.

2. Clustering on Breast Cancer Dataset:
   - The Breast Cancer dataset is split into training and test sets using train_test_split.
   - GMM clustering is performed on the training set.
   - The resulting clusters are used to predict the labels for both the training and test datasets.
   - The accuracy scores are computed by comparing the predicted labels with the true labels.

### Evaluation and Results
1. Accuracy Comparison:
   - The accuracy scores of K-means and GMM clustering algorithms are compared for both datasets.
   - The training and test accuracies are calculated and analyzed.
   - The accuracy scores indicate the effectiveness of each clustering algorithm in correctly assigning data points to clusters.

2. Confusion Matrices:
   - Confusion matrices are created to visualize the clustering results for both the training and test datasets.
   - Heatmaps are generated using the seaborn library to represent the confusion matrices.
  

 - The confusion matrices provide insights into the clustering performance and misclassifications.

### Conclusion
- The analysis of K-means and GMM clustering algorithms on the Heart Disease and Breast Cancer datasets demonstrates their effectiveness in clustering tabular data.
- GMM shows higher accuracy compared to K-means for both datasets, indicating its potential for improved clustering performance.
- The use of confusion matrices enhances the evaluation of clustering algorithms by providing a visual representation of the clustering results and misclassifications.

This repository serves as a detailed example of how to implement and compare clustering algorithms on tabular datasets. The provided code, datasets, and analysis can be utilized as a reference for similar clustering analysis tasks, facilitating the understanding and application of these algorithms in different data scenarios. Contributions, suggestions, and further exploration of the code and datasets are encouraged to expand the project's scope and potential applications.
