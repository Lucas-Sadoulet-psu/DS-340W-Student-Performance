import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

# Establishing a random seed for reproducibility
random.seed(10)
np.random.seed(10)


# loading the dataset
try:
	data_math = pd.read_csv('supporting_paper_1\student-mat.csv')
except Exception as e:
	print("Reading the CSV file resulted in an error:")
	print(str(e))

# Creating a new binary target variable 'Result' based on the final grade 'G3'
data_math['Result'] = ['Pass' if x > 10 else 'Fail' for x in data_math['G3']]
data_math = data_math.drop(columns='G3', axis=1)


# Printing the number of columns, names of columns with the first few rows of the dataset
print(data_math.columns)
print(data_math.head())
print(data_math.shape[1])


# Identify columns that are not numeric
non_numerical_cols = data_math.select_dtypes(include=['object']).columns

# Encoding non-numerical columns with one-hot encoding
data_mathf = pd.get_dummies(data_math, columns=non_numerical_cols, drop_first=True)


# Printing the encoded Dataset
print(data_mathf)
print(data_mathf.shape[1])


# 'X' represents the input features, 'Y' denotes the objective variable
X = data_mathf.drop('Result_Pass', axis=1)
Y = data_mathf['Result_Pass']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Machine learning models by k-Fold Cross Validation
# Creating an empty list in which to store the accuracy values for every iteration
accuracy_values_log_reg = [] # Reserved space for Logistic Regression accuracy values
accuracy_values_log_reg_lasso = [] # Reserved space for Logistic Regression with Lasso accuracy values
accuracy_values_svm = [] # Reserved space for SVM accuracy values
accuracy_values_svm_lasso = [] # Reserved space for SVM with Lasso accuracy values
accuracy_values_knn = [] # Reserved space for KNN accuracy values
accuracy_values_knn_lasso = [] # Reserved space for KNN with Lasso accuracy values
accuracy_values_lin_reg = [] # Reserved space for Linear Regression accuracy values
accuracy_values_lin_reg_lasso = [] # Reserved space for Linear Regression with Lasso accuracy values
accuracy_values_dt = [] # Reserved space for Decision Tree accuracy values
accuracy_values_dt_lasso = [] # Reserved space for Decision Tree with Lasso accuracy values
accuracy_values_rf = [] # Reserved space for Random Forest accuracy values
accuracy_values_rf_lasso = [] # Reserved space for Random Forest with Lasso accuracy values
accuracy_values_xgb = [] # Reserved space for XGBoost accuracy values
accuracy_values_xgb_lasso = [] # Reserved space for XGBoost with Lasso accuracy values
accuracy_values_nn = [] # Reserved space for Neural Network accuracy values
accuracy_values_nn_lasso = [] # Reserved space for Neural Network with Lasso accuracy values

num_iterations = 5 # Number of iterations
n_splits = 5 # Number of splits


for iteration in range(num_iterations):
	
	print('Iteration:', iteration + 1)
	# K-fold splitting
	rkf = KFold(n_splits=n_splits, random_state=iteration, shuffle=True)
	split_sizes = [len(split) for train_idx, split in rkf.split(X_scaled)]
	


	# Lists containing accuracy values for the current iteration
	accuracy_log_reg = []
	accuracy_log_reg_lasso = []
	accuracy_svm = []
	accuracy_svm_lasso = []
	accuracy_knn = []
	accuracy_knn_lasso = []
	accuracy_lin_reg = []
	accuracy_lin_reg_lasso = []
	accuracy_dt = []
	accuracy_dt_lasso = []
	accuracy_rf = []
	accuracy_rf_lasso = []
	accuracy_xgb = []
	accuracy_xgb_lasso = []
	accuracy_nn = []
	accuracy_nn_lasso = []

	# Performing K-fold cross-validation
	for train_idx, test_idx in rkf.split(X_scaled):
		X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
		y_train, y_test = Y[train_idx], Y[test_idx]

		# Initializing models without Lasso
		log_reg = LogisticRegression()
		svm = SVC()
		knn = KNeighborsClassifier()
		dt = DecisionTreeClassifier()
		rf = RandomForestClassifier()
		lin_reg = LinearRegression()
		xgb_model = XGBClassifier()
		tf.random.set_seed(iteration)

		# Creating a list of potential alpha (Lasso penalty) values to test
		alphas = np.logspace(-2, -1, 10)

		# Initializing an array to store the cross-validated scores for each alpha
		cv_scores = []

		lr = LinearRegression()
		# Looping through the alphas
		for i in alphas:
			sel_ = SelectFromModel(Lasso(alpha=i, random_state=iteration))
			sel_.fit(X_train, y_train)
			X_train_selected = sel_.transform(X_train)

			# Fitting a linear regression model
			lr.fit(X_train_selected, y_train)

			# Performing cross-validation on the linear regression model
			scores = cross_val_score(lr, X_train_selected, y_train, cv=5, scoring='neg_mean_squared_error')

			# Storing the mean squared error (MSE) scores
			cv_scores.append(-np.mean(scores))

		# Finding the alpha with the best cross-validated score
		best_alpha = alphas[np.argmin(cv_scores)]

		# Printing the best amount of alpha
		print(best_alpha)

		# Definition of Lasso
		sel_ = SelectFromModel(Lasso(alpha=best_alpha, random_state=iteration))
		sel_.fit(X_train, y_train)
		X_train_selected = sel_.transform(X_train)
		X_test_selected = sel_.transform(X_test)

		log_reg_lasso = LogisticRegression()
		svm_lasso = SVC()
		knn_lasso = KNeighborsClassifier()
		dt_lasso = DecisionTreeClassifier()
		rf_lasso = RandomForestClassifier()
		lin_reg_lasso = LinearRegression()
		xgb_model_lasso = XGBClassifier()

		# Logistic Regression without Lasso
		log_reg.fit(X_train, y_train)
		y_pred_log = log_reg.predict(X_test)
		accuracy_log = accuracy_score(y_test, y_pred_log)
		accuracy_log_reg.append(accuracy_log)

		# Logistic Regression with Lasso feature selection
		log_reg_lasso.fit(X_train_selected, y_train)
		y_pred_log_lasso = log_reg_lasso.predict(X_test_selected)
		accuracy_log_lasso = accuracy_score(y_test, y_pred_log_lasso)
		accuracy_log_reg_lasso.append(accuracy_log_lasso)

		# SVM without Lasso
		svm.fit(X_train, y_train)
		y_pred_svm = svm.predict(X_test)
		accuracy_s = accuracy_score(y_test, y_pred_svm)
		accuracy_svm.append(accuracy_s)

		# SVM with Lasso feature selection
		svm_lasso.fit(X_train_selected, y_train)
		y_pred_svm_lasso = svm_lasso.predict(X_test_selected)
		accuracy_svm_lasso_ = accuracy_score(y_test, y_pred_svm_lasso)
		accuracy_svm_lasso.append(accuracy_svm_lasso_)

		# KNN without Lasso
		knn.fit(X_train, y_train)
		y_pred_knn = knn.predict(X_test)
		accuracy_k = accuracy_score(y_test, y_pred_knn)
		accuracy_knn.append(accuracy_k)

		# KNN with Lasso feature selection
		knn_lasso.fit(X_train_selected, y_train)
		y_pred_knn_lasso = knn_lasso.predict(X_test_selected)
		accuracy_knn_lasso_ = accuracy_score(y_test, y_pred_knn_lasso)
		accuracy_knn_lasso.append(accuracy_knn_lasso_)

		# Linear Regression without Lasso
		lin_reg.fit(X_train, y_train)
		y_pred_lin_reg = lin_reg.predict(X_test)
		y_pred_lin_reg = np.round(y_pred_lin_reg)
		accuracy_lin_reg_ = accuracy_score(y_test, y_pred_lin_reg)
		accuracy_lin_reg.append(accuracy_lin_reg_)

		# Linear Regression with Lasso
		lin_reg_lasso.fit(X_train_selected, y_train)
		y_pred_lin_reg_lasso = lin_reg_lasso.predict(X_test_selected)
		y_pred_lin_reg_lasso = np.round(y_pred_lin_reg_lasso)
		accuracy_lin_reg_lasso_ = accuracy_score(y_test, y_pred_lin_reg_lasso)
		accuracy_lin_reg_lasso.append(accuracy_lin_reg_lasso_)

		# Decision Tree without Lasso
		dt.fit(X_train, y_train)
		y_pred_dt = dt.predict(X_test)
		accuracy_dt_ = accuracy_score(y_test, y_pred_dt)
		accuracy_dt.append(accuracy_dt_)

		# Decision Tree with Lasso
		dt_lasso.fit(X_train_selected, y_train)
		y_pred_dt_lasso = dt_lasso.predict(X_test_selected)
		accuracy_dt_lasso_ = accuracy_score(y_test, y_pred_dt_lasso)
		accuracy_dt_lasso.append(accuracy_dt_lasso_)

		# Random Forest without Lasso
		rf.fit(X_train, y_train)
		y_pred_rf = rf.predict(X_test)
		accuracy_rf_ = accuracy_score(y_test, y_pred_rf)
		accuracy_rf.append(accuracy_rf_)

		# Random Forest with Lasso
		rf_lasso.fit(X_train_selected, y_train)
		y_pred_rf_lasso = rf_lasso.predict(X_test_selected)
		accuracy_rf_lasso_ = accuracy_score(y_test, y_pred_rf_lasso)
		accuracy_rf_lasso.append(accuracy_rf_lasso_)

		# XGBoost without Lasso
		xgb_model.fit(X_train, y_train)
		y_pred_xgb = xgb_model.predict(X_test)
		accuracy_xgb_ = accuracy_score(y_test, y_pred_xgb)
		accuracy_xgb.append(accuracy_xgb_)

		# XGBoost with Lasso feature selection
		xgb_model_lasso.fit(X_train_selected, y_train)
		y_pred_xgb_lasso = xgb_model_lasso.predict(X_test_selected)
		accuracy_xgb_lasso_ = accuracy_score(y_test, y_pred_xgb_lasso)
		accuracy_xgb_lasso.append(accuracy_xgb_lasso_)

		# Neural Network without Lasso
		nn_model = keras.Sequential([
			layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
			layers.Dropout(0.3),
			layers.Dense(64, activation='relu'),
			layers.Dropout(0.3),
			layers.Dense(32, activation='relu'),
			layers.Dropout(0.2),
			layers.Dense(1, activation='sigmoid')
		])
		nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		nn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
		y_pred_nn = (nn_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
		accuracy_nn_ = accuracy_score(y_test, y_pred_nn)
		accuracy_nn.append(accuracy_nn_)

		# Neural Network with Lasso feature selection
		nn_model_lasso = keras.Sequential([
			layers.Dense(64, activation='relu', input_shape=(X_train_selected.shape[1],)),
			layers.Dropout(0.3),
			layers.Dense(32, activation='relu'),
			layers.Dropout(0.2),
			layers.Dense(1, activation='sigmoid')
		])
		nn_model_lasso.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		nn_model_lasso.fit(X_train_selected, y_train, epochs=50, batch_size=16, verbose=0)
		y_pred_nn_lasso = (nn_model_lasso.predict(X_test_selected, verbose=0) > 0.5).astype(int).flatten()
		accuracy_nn_lasso_ = accuracy_score(y_test, y_pred_nn_lasso)
		accuracy_nn_lasso.append(accuracy_nn_lasso_)

	accuracy_values_log_reg.append(sum(accuracy_log_reg) / n_splits)
	accuracy_values_log_reg_lasso.append(sum(accuracy_log_reg_lasso) / n_splits)
	accuracy_values_lin_reg.append(sum(accuracy_lin_reg) / n_splits)
	accuracy_values_lin_reg_lasso.append(sum(accuracy_lin_reg_lasso) / n_splits)
	accuracy_values_dt.append(sum(accuracy_dt) / n_splits)
	accuracy_values_dt_lasso.append(sum(accuracy_dt_lasso) / n_splits)
	accuracy_values_rf.append(sum(accuracy_rf) / n_splits)
	accuracy_values_rf_lasso.append(sum(accuracy_rf_lasso) / n_splits)
	accuracy_values_svm.append(sum(accuracy_svm) / n_splits)
	accuracy_values_svm_lasso.append(sum(accuracy_svm_lasso) / n_splits)
	accuracy_values_knn.append(sum(accuracy_knn) / n_splits)
	accuracy_values_knn_lasso.append(sum(accuracy_knn_lasso) / n_splits)
	accuracy_values_xgb.append(sum(accuracy_xgb) / n_splits)
	accuracy_values_xgb_lasso.append(sum(accuracy_xgb_lasso) / n_splits)
	accuracy_values_nn.append(sum(accuracy_nn) / n_splits)
	accuracy_values_nn_lasso.append(sum(accuracy_nn_lasso) / n_splits)



# Calculating mean accuracy values
mean_accuracy_log_reg = np.mean(accuracy_values_log_reg)
mean_accuracy_log_reg_lasso = np.mean(accuracy_values_log_reg_lasso)
mean_accuracy_svm = np.mean(accuracy_values_svm)
mean_accuracy_svm_lasso = np.mean(accuracy_values_svm_lasso)
mean_accuracy_knn = np.mean(accuracy_values_knn)
mean_accuracy_knn_lasso = np.mean(accuracy_values_knn_lasso)
mean_accuracy_lin_reg = np.mean(accuracy_values_lin_reg)
mean_accuracy_lin_reg_lasso = np.mean(accuracy_values_lin_reg_lasso)
mean_accuracy_dt = np.mean(accuracy_values_dt)
mean_accuracy_dt_lasso = np.mean(accuracy_values_dt_lasso)
mean_accuracy_rf = np.mean(accuracy_values_rf)
mean_accuracy_rf_lasso = np.mean(accuracy_values_rf_lasso)
mean_accuracy_xgb = np.mean(accuracy_values_xgb)
mean_accuracy_xgb_lasso = np.mean(accuracy_values_xgb_lasso)
mean_accuracy_nn = np.mean(accuracy_values_nn)
mean_accuracy_nn_lasso = np.mean(accuracy_values_nn_lasso)

# Calculating mean accuracy values as percentages
mean_accuracy_log_reg = np.mean(accuracy_values_log_reg) * 100
mean_accuracy_log_reg_lasso = np.mean(accuracy_values_log_reg_lasso) * 100
mean_accuracy_svm = np.mean(accuracy_values_svm) * 100
mean_accuracy_svm_lasso = np.mean(accuracy_values_svm_lasso) * 100
mean_accuracy_knn = np.mean(accuracy_values_knn) * 100
mean_accuracy_knn_lasso = np.mean(accuracy_values_knn_lasso) * 100
mean_accuracy_lin_reg = np.mean(accuracy_values_lin_reg) * 100
mean_accuracy_lin_reg_lasso = np.mean(accuracy_values_lin_reg_lasso) * 100
mean_accuracy_dt = np.mean(accuracy_values_dt) * 100
mean_accuracy_dt_lasso = np.mean(accuracy_values_dt_lasso) * 100
mean_accuracy_rf = np.mean(accuracy_values_rf) * 100
mean_accuracy_rf_lasso = np.mean(accuracy_values_rf_lasso) * 100
mean_accuracy_xgb = np.mean(accuracy_values_xgb) * 100
mean_accuracy_xgb_lasso = np.mean(accuracy_values_xgb_lasso) * 100
mean_accuracy_nn = np.mean(accuracy_values_nn) * 100
mean_accuracy_nn_lasso = np.mean(accuracy_values_nn_lasso) * 100

# Printing mean accuracy values as percentages
print("Mean Accuracy Values Without Lasso:")
print("Logistic Regression: {:.2f}%".format(mean_accuracy_log_reg))
print("Logistic Regression with Lasso: {:.2f}%".format(mean_accuracy_log_reg_lasso))
print("SVM: {:.2f}%".format(mean_accuracy_svm))
print("SVM with Lasso: {:.2f}%".format(mean_accuracy_svm_lasso))
print("KNN: {:.2f}%".format(mean_accuracy_knn))
print("KNN with Lasso: {:.2f}%".format(mean_accuracy_knn_lasso))
print("Linear Regression: {:.2f}%".format(mean_accuracy_lin_reg))
print("Linear Regression with Lasso: {:.2f}%".format(mean_accuracy_lin_reg_lasso))
print("Decision Tree: {:.2f}%".format(mean_accuracy_dt))
print("Decision Tree with Lasso: {:.2f}%".format(mean_accuracy_dt_lasso))
print("Random Forest: {:.2f}%".format(mean_accuracy_rf))
print("Random Forest with Lasso: {:.2f}%".format(mean_accuracy_rf_lasso))
print("XGBoost: {:.2f}%".format(mean_accuracy_xgb))
print("XGBoost with Lasso: {:.2f}%".format(mean_accuracy_xgb_lasso))
print("Neural Network: {:.2f}%".format(mean_accuracy_nn))
print("Neural Network with Lasso: {:.2f}%".format(mean_accuracy_nn_lasso))

# Creating a boxplot for machine learning models with Lasso for K-Fold Cross Validation
plt.figure(figsize=(12, 6))
accuracy_values_with_lasso_kfold = [accuracy_values_log_reg_lasso, accuracy_values_svm_lasso,
								accuracy_values_knn_lasso, accuracy_values_lin_reg_lasso,
								accuracy_values_dt_lasso, accuracy_values_rf_lasso, accuracy_values_xgb_lasso,
								accuracy_values_nn_lasso]
labels_with_lasso_kfold = ['Logistic Regression with Lasso', 'SVM with Lasso', 'KNN with Lasso',
					  'Linear Regression with Lasso', 'Decision Tree with Lasso',
					  'Random Forest with Lasso', 'XGBoost with Lasso', 'Neural Network with Lasso']
plt.boxplot(accuracy_values_with_lasso_kfold, labels=labels_with_lasso_kfold, vert=False, showmeans=True)
plt.xlabel('Accuracy')
plt.title('Machine Learning with Lasso - k-Fold Cross Validation', fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Creating a boxplot for machine learning models without Lasso for K-Fold Cross Validation
plt.figure(figsize=(12, 6))
accuracy_values_without_lasso_kfold = [accuracy_values_log_reg, accuracy_values_svm,
								   accuracy_values_knn, accuracy_values_lin_reg,
								   accuracy_values_dt, accuracy_values_rf, accuracy_values_xgb,
								   accuracy_values_nn]
labels_without_lasso_kfold = ['Logistic Regression', 'SVM', 'KNN', 'Linear Regression',
						 'Decision Tree', 'Random Forest', 'XGBoost', 'Neural Network']
plt.boxplot(accuracy_values_without_lasso_kfold, labels=labels_without_lasso_kfold, vert=False, showmeans=True)
plt.xlabel('Accuracy')
plt.title('Machine Learning without Lasso - k-Fold Cross Validation', fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyzing the impact of parental jobs on student outcomes
parental_job_cols = ['Mjob', 'Fjob']  # Mother's job and Father's job

# Reload original data to access job columns (before dropping G3)
data_analysis = pd.read_csv('supporting_paper_1\student-mat.csv')
data_analysis['Result'] = ['Pass' if x > 10 else 'Fail' for x in data_analysis['G3']]

print("\n=== Parental Job Impact Analysis ===")

# Analyze pass rate by mother's job
print("\nPass Rate by Mother's Job:")
mother_job_analysis = data_analysis.groupby('Mjob')['Result'].value_counts(normalize=True).unstack()
print(mother_job_analysis)

# Analyze pass rate by father's job
print("\nPass Rate by Father's Job:")
father_job_analysis = data_analysis.groupby('Fjob')['Result'].value_counts(normalize=True).unstack()
print(father_job_analysis)

# Analyze primary guardian impact
print("\n=== Primary Guardian Impact Analysis ===")
print("\nGrade Distribution by Primary Guardian:")
guardian_analysis = data_analysis.groupby('guardian')['G3'].describe()
print(guardian_analysis)

print("\nPass Rate by Primary Guardian:")
guardian_pass_analysis = data_analysis.groupby('guardian')['Result'].value_counts(normalize=True).unstack()
print(guardian_pass_analysis)

# Create a column for primary parent's job based on guardian type
print("\n=== Primary Parent's Job Impact Analysis ===")
data_analysis['primary_parent_job'] = data_analysis.apply(
	lambda row: row['Mjob'] if row['guardian'] == 'mother' else (row['Fjob'] if row['guardian'] == 'father' else 'other'),
	axis=1
)

print("\nGrade Distribution by Primary Parent's Job:")
primary_parent_job_analysis = data_analysis.groupby('primary_parent_job')['G3'].describe()
print(primary_parent_job_analysis)

print("\nPass Rate by Primary Parent's Job:")
primary_parent_pass_analysis = data_analysis.groupby('primary_parent_job')['Result'].value_counts(normalize=True).unstack()
print(primary_parent_pass_analysis)

# Visualize the impact using box plots instead of bar plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor('white')

# Define color palettes for each plot
job_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
colors_guardian = ['#FF7675', '#74B9FF', '#A29BFE']

# Box plot for mother's job impact on grades
bp1 = axes[0, 0].boxplot([data_analysis[data_analysis['Mjob'] == job]['G3'].values for job in sorted(data_analysis['Mjob'].unique())],
						  labels=sorted(data_analysis['Mjob'].unique()),
						  patch_artist=True,
						  widths=0.6,
						  showmeans=True,
						  meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=5))

for patch, color in zip(bp1['boxes'], job_colors):
	patch.set_facecolor(color)
	patch.set_alpha(0.7)
	
for whisker in bp1['whiskers']:
	whisker.set(linewidth=1.5, color='#2C3E50')
	
for median in bp1['medians']:
	median.set(linewidth=2.5, color='#C0392B')

axes[0, 0].set_title('Grade Distribution by Mother\'s Job', fontweight='bold', fontsize=13, pad=15)
axes[0, 0].set_ylabel('Final Grade (G3)', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Mother\'s Job', fontsize=11, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45, labelsize=10)
axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0, 0].set_facecolor('#F8F9FA')

# Box plot for father's job impact on grades
bp2 = axes[0, 1].boxplot([data_analysis[data_analysis['Fjob'] == job]['G3'].values for job in sorted(data_analysis['Fjob'].unique())],
	labels=sorted(data_analysis['Fjob'].unique()),
	patch_artist=True,
	widths=0.6,
	showmeans=True,
	meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=5))

for patch, color in zip(bp2['boxes'], job_colors):
	patch.set_facecolor(color)
	patch.set_alpha(0.7)
	
for whisker in bp2['whiskers']:
	whisker.set(linewidth=1.5, color='#2C3E50')
	
for median in bp2['medians']:
	median.set(linewidth=2.5, color='#C0392B')

axes[0, 1].set_title('Grade Distribution by Father\'s Job', fontweight='bold', fontsize=13, pad=15)
axes[0, 1].set_ylabel('Final Grade (G3)', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Father\'s Job', fontsize=11, fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45, labelsize=10)
axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
axes[0, 1].set_facecolor('#F8F9FA')

# Box plot for primary guardian impact on grades
bp3 = axes[1, 0].boxplot([data_analysis[data_analysis['guardian'] == guardian]['G3'].values for guardian in sorted(data_analysis['guardian'].unique())],
	labels=sorted(data_analysis['guardian'].unique()),
	patch_artist=True,
	widths=0.6,
	showmeans=True,
	meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=5))

for patch, color in zip(bp3['boxes'], colors_guardian):
	patch.set_facecolor(color)
	patch.set_alpha(0.7)
	
for whisker in bp3['whiskers']:
	whisker.set(linewidth=1.5, color='#2C3E50')
	
for median in bp3['medians']:
	median.set(linewidth=2.5, color='#C0392B')

axes[1, 0].set_title('Grade Distribution by Primary Guardian', fontweight='bold', fontsize=13, pad=15)
axes[1, 0].set_ylabel('Final Grade (G3)', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Primary Guardian', fontsize=11, fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45, labelsize=10)
axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
axes[1, 0].set_facecolor('#F8F9FA')

# Box plot for primary parent's job impact on grades
bp4 = axes[1, 1].boxplot([data_analysis[data_analysis['primary_parent_job'] == job]['G3'].values for job in sorted(data_analysis['primary_parent_job'].unique())],
	labels=sorted(data_analysis['primary_parent_job'].unique()),
	patch_artist=True,
	widths=0.6,
	showmeans=True,
	meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=5))

for patch, color in zip(bp4['boxes'], job_colors):
	patch.set_facecolor(color)
	patch.set_alpha(0.7)
	
for whisker in bp4['whiskers']:
	whisker.set(linewidth=1.5, color='#2C3E50')
	
for median in bp4['medians']:
	median.set(linewidth=2.5, color='#C0392B')

axes[1, 1].set_title('Grade Distribution by Primary Parent\'s Job', fontweight='bold', fontsize=13, pad=15)
axes[1, 1].set_ylabel('Final Grade (G3)', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Primary Parent\'s Job', fontsize=11, fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45, labelsize=10)
axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
axes[1, 1].set_facecolor('#F8F9FA')

fig.suptitle('Impact of Parental Factors on Student Academic Performance', 
			 fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.show()

# ========================= FEATURE IMPORTANCE ANALYSIS =========================
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS - Most Impactful Parameters to Student Success")
print("="*80)

# Train final models to extract feature importance
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_scaled, Y, test_size=0.2, random_state=10)

# Dictionary to store feature importances from each model
feature_importances = {}

# 1. Random Forest Feature Importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=10)
rf_model.fit(X_train_final, y_train_final)
feature_importances['Random Forest'] = rf_model.feature_importances_

# 2. XGBoost Feature Importance
xgb_model = XGBClassifier(random_state=10, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_final, y_train_final)
feature_importances['XGBoost'] = xgb_model.feature_importances_

# 3. Decision Tree Feature Importance
dt_model = DecisionTreeClassifier(random_state=10)
dt_model.fit(X_train_final, y_train_final)
feature_importances['Decision Tree'] = dt_model.feature_importances_

# 4. Logistic Regression Coefficients (absolute values)
lr_model = LogisticRegression(max_iter=1000, random_state=10)
lr_model.fit(X_train_final, y_train_final)
feature_importances['Logistic Regression'] = np.abs(lr_model.coef_[0])

# Get feature names
feature_names = X.columns.tolist()

# Create a comprehensive visualization of feature importance
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor('white')

models = ['Random Forest', 'XGBoost', 'Decision Tree', 'Logistic Regression']
colors_importance = ['#5B9BD5', '#5B9BD5', '#5B9BD5', '#5B9BD5']  # Same blue color for all

for idx, (model_name, ax) in enumerate(zip(models, axes.flat)):
	importances = feature_importances[model_name]
	
	# Get top 15 features
	top_indices = np.argsort(importances)[-15:][::-1]
	top_features = [feature_names[i] for i in top_indices]
	top_importances = importances[top_indices]
	
	# Create bar plot
	bars = ax.barh(range(len(top_features)), top_importances, color=colors_importance[idx], alpha=0.8, edgecolor='black', linewidth=1.2)
	ax.set_yticks(range(len(top_features)))
	ax.set_yticklabels(top_features, fontsize=8)
	ax.set_xlabel('Importance Score', fontsize=9, fontweight='bold')
	ax.set_title(f'{model_name}\nTop 15 Features for Student Success', fontsize=10, fontweight='bold', pad=10)
	ax.grid(axis='x', alpha=0.3, linestyle='--')
	ax.set_facecolor('#F8F9FA')
	
	# Add value labels on bars
	for i, (bar, val) in enumerate(zip(bars, top_importances)):
		ax.text(val, i, f' {val:.4f}', va='center', fontsize=7)
	
	# Print top features for this model
	print(f"\n{model_name} - Top 15 Most Impactful Features:")
	print("-" * 60)
	for i, (feat, imp) in enumerate(zip(top_features, top_importances), 1):
		print(f"  {i:2d}. {feat:40s} - Score: {imp:.6f}")

fig.suptitle('Feature Importance Across All Machine Learning Models', 
			 fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()
plt.show()

# ========================= AGGREGATE IMPORTANCE ANALYSIS =========================
print("\n" + "="*80)
print("AGGREGATE IMPORTANCE RANKING - Combined Insights from All Models")
print("="*80)

# Normalize importances to 0-1 scale for each model
normalized_importances = {}
for model_name, importances in feature_importances.items():
	max_imp = np.max(importances)
	if max_imp > 0:
		normalized_importances[model_name] = importances / max_imp
	else:
		normalized_importances[model_name] = importances

# Calculate average importance across all models
avg_importance = np.zeros(len(feature_names))
for model_importances in normalized_importances.values():
	avg_importance += model_importances
avg_importance /= len(normalized_importances)

# Get top 15 overall features
top_overall_indices = np.argsort(avg_importance)[-15:][::-1]
top_overall_features = [feature_names[i] for i in top_overall_indices]
top_overall_importance = avg_importance[top_overall_indices]

# Visualize aggregate importance
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')

bars = ax.barh(range(len(top_overall_features)), top_overall_importance, 
			   color='#5B9BD5', alpha=0.85, edgecolor='black', linewidth=1.5)
ax.set_yticks(range(len(top_overall_features)))
ax.set_yticklabels(top_overall_features, fontsize=9, fontweight='bold')
ax.set_xlabel('Average Importance Score (Normalized)', fontsize=10, fontweight='bold')
ax.set_title('Top 15 Most Impactful Parameters for Student Success\n(Aggregate Across All Models)', 
			 fontsize=12, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_facecolor('#F8F9FA')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_overall_importance)):
	ax.text(val, i, f' {val:.4f}', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nTop 15 Overall Most Impactful Features for Student Success:")
print("-" * 60)
for i, (feat, imp) in enumerate(zip(top_overall_features, top_overall_importance), 1):
	print(f"  {i:2d}. {feat:40s} - Avg Score: {imp:.6f}")
