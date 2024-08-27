### Project Overview: Machine Learning with MATLAB for Classification of Human Activities

**Objective:**  
The primary goal of this project is to develop, evaluate, and implement a machine learning model to classify human activities based on sensor data collected from a smartphone. This involves the use of MATLAB to preprocess data, train models, and deploy the final model for real-time classification of activities such as sitting, standing, lying down, walking, and jogging.

---
**Part III: Model Evaluation & Implementation**
   - **Recap: Training the Model on Prep Data and Target Data (Optional)**
   - **Model Evaluation**
     - Confusion Matrix
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - Task 1
   - **Model Improvement with Classification Learner App**
     - Feature Selection
     - Task 2
     - Dimensionality Reduction with Principal Component Analysis (PCA)
     - Task 3
   - **Model Implementation**
     - Stream Live Data with MATLAB Mobile
     - Collect Live Data and Predict the Label
     - Task 4

---

### 1. Introduction

#### Use Case: Classification of Human Activities

This project focuses on developing a machine learning model to classify human activities using sensor data from a smartphone's accelerometer and gyroscope. The activities to be classified include sitting, standing, lying down, walking, and jogging.

#### Workflow

The machine learning workflow involves eight key steps:

1. **Data Acquisition:** Collecting raw sensor data from the smartphone.
2. **Data Preprocessing:** Cleaning and normalizing the data.
3. **Feature Engineering:** Extracting meaningful features from the data.
4. **Model Training:** Training various classification algorithms.
5. **Hyperparameter Optimization:** Fine-tuning the model for better performance.
6. **Model Evaluation:** Assessing the model's accuracy and other performance metrics.
7. **Model Improvement:** Enhancing the model using techniques like feature selection and dimensionality reduction.
8. **Integration:** Implementing the final model for real-time data classification.

---

### 2. Part III: Model Evaluation & Implementation

#### Recap: Training the Model on Prep Data and Target Data (Optional)

To get started, load the preprocessed data from Part I and the trained classifiers from Part II. If the classifiers cannot be loaded, you can recreate them using the provided code.

```matlab
clear all

% Load data
load raw_data.csv     % raw data 6 feature variables 1 label
load prep_data.csv    % data cleaned and normalized 6 feature variables 1 label
load target_data.csv  % data with sliding window 

% Split data into training and test set
rng('default') % For reproducibility
n = length(prep_data);

hpartition = cvpartition(n,'Holdout',0.15); % Nonstratified partition

idxTrain = training(hpartition);
Train_prep = prep_data(idxTrain,:);

idxNew = test(hpartition);
Test_prep = prep_data(idxNew,:);

% Train classifier and optimize its hyperparameters
Mdl_bay_prep = fitctree(Train_prep(:,1:6), Train_prep(:,7),'OptimizeHyperparameters','all',...
    'HyperparameterOptimizationOptions',...
    struct('Optimizer','bayesopt'))

% Predict the label for new data
predictedY = predict(Mdl_bay_prep,Test_prep(:,1:6))
```

#### Model Evaluation

Evaluate the trained models using a confusion matrix and compute accuracy, precision, recall, and F1-score.

##### Confusion Matrix

```matlab
C = confusionmat(Test_prep(:,7),predictedY)
cm = confusionchart(C);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
```

##### Accuracy, Precision, Recall, F1-Score

Accuracy can be calculated by summing the true positives and dividing by the total number of cases.

```matlab
% Calculate accuracy, precision, recall, and F1-score
function score = evaluation(C)
    precision = diag(C)./sum(C,1)';
    recall = diag(C)./sum(C,2);
    f1score = 2*(precision.*recall)./(precision+recall);
    score = table(precision, recall, f1score);
end

score = evaluation(C);
f1macro = mean(score.f1score);
```

##### Task 1: Model Evaluation

1. Train classifiers on `prep_data` and predict values on the test set.
2. Create the confusion matrix with row and column summations.
3. Calculate the accuracy, recall, precision, and F1-score for each class.
4. Compute the macro F1-score and compare it with the results from Part II.

#### Model Improvement with Classification Learner App

Enhance the model's performance using feature selection and PCA for dimensionality reduction.

##### Task 2: Feature Selection

1. Load `target_data` into the Classification Learner App.
2. Use different ranking algorithms to select a subset of features.
3. Train the model and assess the impact of feature reduction.

##### Task 3: Dimensionality Reduction with PCA

1. Enable PCA in the Classification Learner App.
2. Train models with PCA applied and compare the results to the original models.

#### Model Implementation

After evaluating and improving the model, implement it for real-time classification using live data streamed from the smartphone.

##### Stream Live Data with MATLAB Mobile

Set up the MATLAB Mobile App to stream live sensor data.

```matlab
% Establish connection to the mobile device
m = mobiledev("YourDeviceName");
m.Logging = 1;

% Collect and preprocess live data
while m.Logging == 1
    if length(accellog(m)) >= 128 && length(angvellog(m)) >= 128 
        acc = accellog(m); 
        ang = angvellog(m);
        A = horzcat(acc((length(acc)-127):length(acc),:), ang((length(ang)-127):length(ang),:));
        A = normalize(A);
        data = horzcat(mean(A), std(A), min(A), max(A));
        predict(Mdl_bay,data) % Predict the new label
        pause(2);
    else
        pause(1);
    end
end
disp('End')
```

##### Task 4: Real-Time Data Classification

1. Set up a sensor live stream from the MATLAB Mobile App.
2. Adjust the provided code to fit your preprocessing settings.
3. Use your best-performing classifier to predict activities based on real-time data.

---

### Conclusion

This project provides a comprehensive workflow for developing a machine learning model using MATLAB to classify human activities based on sensor data. The project not only covers the theoretical aspects of model evaluation and improvement but also emphasizes practical implementation by streaming and classifying live data in real-time.
