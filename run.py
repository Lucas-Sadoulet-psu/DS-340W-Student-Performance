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

num_iterations = 10 # Number of iterations
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

# Creating a boxplot for machine learning models with Lasso for K-Fold Cross Validation
plt.figure(figsize=(12, 6))
accuracy_values_with_lasso_kfold = [accuracy_values_log_reg_lasso, accuracy_values_svm_lasso,
									accuracy_values_knn_lasso, accuracy_values_lin_reg_lasso,
									accuracy_values_dt_lasso, accuracy_values_rf_lasso, accuracy_values_xgb_lasso]
labels_with_lasso_kfold = ['Logistic Regression with Lasso', 'SVM with Lasso', 'KNN with Lasso',
						  'Linear Regression with Lasso', 'Decision Tree with Lasso',
						  'Random Forest with Lasso', 'XGBoost with Lasso']

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
									   accuracy_values_dt, accuracy_values_rf, accuracy_values_xgb]
labels_without_lasso_kfold = ['Logistic Regression', 'SVM', 'KNN', 'Linear Regression',
							 'Decision Tree', 'Random Forest', 'XGBoost']

plt.boxplot(accuracy_values_without_lasso_kfold, labels=labels_without_lasso_kfold, vert=False, showmeans=True)
plt.xlabel('Accuracy')
plt.title('Machine Learning without Lasso - k-Fold Cross Validation', fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()