import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import VotingClassifier

# Load the accelerometer data from a CSV file
accelerometer_data = pd.read_csv('accelerometer_data.csv')  # Replace with your CSV file name

# Split the accelerometer data into features (X_accel) and labels (y_accel)
X_accel = accelerometer_data.drop('label', axis=1)
y_accel = accelerometer_data['label']

# Load the gyroscope data from a CSV file
gyroscope_data = pd.read_csv('gyroscope_data.csv')  # Replace with your CSV file name

# Split the gyroscope data into features (X_gyro) and labels (y_gyro)
X_gyro = gyroscope_data.drop('label', axis=1)
y_gyro = gyroscope_data['label']

# Split the accelerometer and gyroscope data into training and testing sets
X_train_accel, X_test_accel, y_train_accel, y_test_accel = train_test_split(X_accel, y_accel, test_size=0.2, random_state=42)
X_train_gyro, X_test_gyro, y_train_gyro, y_test_gyro = train_test_split(X_gyro, y_gyro, test_size=0.2, random_state=42)

# Create an imputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Apply the imputer to both accelerometer and gyroscope data
X_train_accel = imputer.fit_transform(X_train_accel)
X_test_accel = imputer.transform(X_test_accel)

X_train_gyro = imputer.fit_transform(X_train_gyro)
X_test_gyro = imputer.transform(X_test_gyro)

# Augment the training data for both accelerometer and gyroscope
oversampler = RandomOverSampler(sampling_strategy={cls: 500 for cls in set(y_train_accel)}, random_state=42)

X_train_accel_oversampled, y_train_accel_oversampled = oversampler.fit_resample(X_train_accel, y_train_accel)
X_train_gyro_oversampled, y_train_gyro_oversampled = oversampler.fit_resample(X_train_gyro, y_train_gyro)

# Create and train individual classifiers for accelerometer data
accel_classifiers = {
    'J48 (Decision Tree)': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', C=1.0),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
}

accel_results = {}
for clf_name, classifier in accel_classifiers.items():
    classifier.fit(X_train_accel_oversampled, y_train_accel_oversampled)
    y_pred_accel = classifier.predict(X_test_accel)

    accuracy_accel = accuracy_score(y_test_accel, y_pred_accel)
    report_accel = classification_report(y_test_accel, y_pred_accel)

    accel_results[clf_name] = {
        'accuracy': accuracy_accel,
        'classification_report': report_accel
    }

# Create and train individual classifiers for gyroscope data
gyro_classifiers = {
    'J48 (Decision Tree)': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', C=1.0),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3)
}

gyro_results = {}
for clf_name, classifier in gyro_classifiers.items():
    classifier.fit(X_train_gyro_oversampled, y_train_gyro_oversampled)
    y_pred_gyro = classifier.predict(X_test_gyro)

    accuracy_gyro = accuracy_score(y_test_gyro, y_pred_gyro)
    report_gyro = classification_report(y_test_gyro, y_pred_gyro)

    gyro_results[clf_name] = {
        'accuracy': accuracy_gyro,
        'classification_report': report_gyro
    }

# Print the results for the accelerometer classifiers
for clf_name, result in accel_results.items():
    print(f'Accelerometer - {clf_name} Accuracy:', result['accuracy'])
    print(f'Accelerometer - {clf_name} Classification Report:\n', result['classification_report'])
    print('------------------------------------------')

# Print the results for the gyroscope classifiers
for clf_name, result in gyro_results.items():
    print(f'Gyroscope - {clf_name} Accuracy:', result['accuracy'])
    print(f'Gyroscope - {clf_name} Classification Report:\n', result['classification_report'])
    print('------------------------------------------')

# Create an ensemble of classifiers for accelerometer data
accel_ensemble = VotingClassifier(estimators=[
    ('Decision Tree', accel_classifiers['J48 (Decision Tree)']),
    ('Random Forest', accel_classifiers['Random Forest']),
    ('SVM', accel_classifiers['SVM']),
    ('Naive Bayes', accel_classifiers['Naive Bayes']),
    ('K-Nearest Neighbors', accel_classifiers['K-Nearest Neighbors'])
], voting='hard')

# Create an ensemble of classifiers for gyroscope data
gyro_ensemble = VotingClassifier(estimators=[
    ('Decision Tree', gyro_classifiers['J48 (Decision Tree)']),
    ('Random Forest', gyro_classifiers['Random Forest']),
    ('SVM', gyro_classifiers['SVM']),
    ('Naive Bayes', gyro_classifiers['Naive Bayes']),
    ('K-Nearest Neighbors', gyro_classifiers['K-Nearest Neighbors'])
], voting='hard')

# Train the accelerometer ensemble on the oversampled training data
accel_ensemble.fit(X_train_accel_oversampled, y_train_accel_oversampled)

# Train the gyroscope ensemble on the oversampled training data
gyro_ensemble.fit(X_train_gyro_oversampled, y_train_gyro_oversampled)

# Make predictions using the accelerometer ensemble
y_pred_accel_ensemble = accel_ensemble.predict(X_test_accel)

# Make predictions using the gyroscope ensemble
y_pred_gyro_ensemble = gyro_ensemble.predict(X_test_gyro)

# Evaluate the ensemble for accelerometer data
accuracy_accel_ensemble = accuracy_score(y_test_accel, y_pred_accel_ensemble)
report_accel_ensemble = classification_report(y_test_accel, y_pred_accel_ensemble)

# Evaluate the ensemble for gyroscope data
accuracy_gyro_ensemble = accuracy_score(y_test_gyro, y_pred_gyro_ensemble)
report_gyro_ensemble = classification_report(y_test_gyro, y_pred_gyro_ensemble)

# Print the results for the accelerometer ensemble
print('Accelerometer Ensemble Accuracy:', accuracy_accel_ensemble)
print('Accelerometer Ensemble Classification Report:\n', report_accel_ensemble)

# Print the results for the gyroscope ensemble
print('Gyroscope Ensemble Accuracy:', accuracy_gyro_ensemble)
print('Gyroscope Ensemble Classification Report:\n', report_gyro_ensemble)