# Machine-Learning-with-Matlab

This project focuses on developing a machine learning model that can classify different human activities based on sensor data from a smartphone. The activities include standing, sitting, walking, and knee raises. By using the sensors in a smartphone, specifically the accelerometer and gyroscope, the project aims to gather, process, and analyze data to train a model capable of distinguishing between these activities.

# Introduction to the Project

The project revolves around a common use case in health monitoring apps where different physical activities are recognized automatically. The goal is to classify activities like sitting, standing, walking, and knee raises using data from a smartphone's sensors. The project follows a structured workflow involving data acquisition, preprocessing, feature engineering, model training, and evaluation.

# Part I: Data Acquisition & Preprocessing

**Data Acquisition:**

The first step is to collect data using the MATLAB Mobile app, which reads various sensor data from your smartphone, such as acceleration and angular velocity. The data is captured by performing different activities while carrying the smartphone in your pocket. For each activity, the sensor data is saved in MATLAB Cloud and later retrieved for analysis.

- **Task 1**: I collected data for activities like standing, walking, and sitting by enabling the accelerometer and angular velocity sensors on your smartphone and then saving this data for further analysis.

**Data Preprocessing:**

Once the data is collected, it undergoes preprocessing to prepare it for feature extraction and model training. This involves handling missing data, synchronizing the data from different sensors, removing outliers, and normalizing the data to ensure consistency and accuracy.

- **Missing Values**: Checked for gaps in the data due to potential interruptions during recording and filled in these gaps where necessary.
- **Synchronization**: Merged data from the accelerometer and gyroscope to ensure that the data points align correctly across the two sensors.
- **Outlier Removal**: Identified and removed any anomalous data points that could distort the model's learning process.
- **Normalization**: Scaled the data so that all features contribute equally to the model.

# Feature Engineering

In this stage, raw sensor data is transformed into a set of meaningful features that better represent the underlying patterns of the activities. This is done using the sliding window technique, which divides the data into overlapping segments (windows) and calculates statistical features like mean, standard deviation, minimum, and maximum for each segment.

- **Sliding Window Method**: Used to break the data into segments, ensuring that each activity is analyzed over a small time frame rather than a single data point.
- **Feature Extraction**: Computed features from each window, which are then used to create a new dataset with these derived features.

# Test Task

To consolidate your understanding and apply the skills learned, a comprehensive test task was completed:

- **Activity Data Collection**: You captured data for four different activities: standing, sitting, walking, and knee raises.
- **Data Labeling and Merging**: Each dataset was labeled based on the activity and merged into a single comprehensive dataset.
- **Preprocessing**: This dataset was cleaned, missing values were handled, outliers were removed, and the data was normalized.
- **Feature Engineering**: Applied the sliding window method to create a final dataset that included features derived from the raw sensor data, which would be used for training the machine learning model.

Finally, the project concluded with saving all datasets and code scripts in a zip folder, encapsulating the entire workflow from data acquisition to feature engineering. This final dataset is ready to be used for training and testing a machine learning model to classify human activities.

This overview captures the essence of the project, focusing on the high-level process without delving into the technical details of the code and functions used.
