def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

from flask import Flask
app = Flask(__name__)

@app.route("/")
def classifier():


	# Load libraries
	import pandas
	from pandas.tools.plotting import scatter_matrix
	import matplotlib.pyplot as plt, mpld3
	from sklearn import model_selection
	from sklearn.metrics import classification_report
	from sklearn.metrics import confusion_matrix
	from sklearn.metrics import accuracy_score
	from sklearn.linear_model import LogisticRegression
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	from sklearn.naive_bayes import GaussianNB
	from sklearn.svm import SVC
	import timeit
	import os



	# Load dataset
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	dataset = pandas.read_csv(url, names=names)

# descriptions
#print(dataset.describe())

# Split-out validation dataset
	array = dataset.values
	X = array[:,0:4]
	Y = array[:,4]
	validation_size = 0.20
	seed = 7
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
	seed = 7
	scoring = 'accuracy'

	# Spot Check Algorithms
	models = []
	models.append(('SVM', SVC()))
	models.append(('KNN', KNeighborsClassifier()))


	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
		start_svm_eval = timeit.default_timer()
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		stop_svm_eval = timeit.default_timer()
		print 'Cross-validation score for',msg
		print'Estimation Time: ',stop_svm_eval-start_svm_eval


	#SVM
	print 'SVM:\n'
	# Make predictions on validation dataset
	start_svm = timeit.default_timer()
	svm = SVC()
	svm.fit(X_train, Y_train)
	predictions = svm.predict(X_validation)

	print 'CONFUSION MATRIX:\n'
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))

	stop_svm = timeit.default_timer()

	print'Prediction Accuracy for SVM:',accuracy_score(Y_validation, predictions)
	print 'Prediction Time(secs): ',stop_svm - start_svm


	#KNN
	print 'KNN:\n'
	# Make predictions on validation dataset
	start_knn = timeit.default_timer()

	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	predictions = knn.predict(X_validation)

	print 'CONFUSION MATRIX:\n'
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))

	stop_knn = timeit.default_timer()

	print'Prediction Accuracy for KNN:',accuracy_score(Y_validation, predictions)
	print 'Prediction Time(secs): ',stop_knn - start_knn


	# Compare Algorithms
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	fig = plt.gcf()
	d3plot = mpld3.fig_to_html(fig, template_type="simple")	


	print 'Memory Utilization: ',memory_usage_psutil()

	return d3plot


