import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model 
from sklearn import cross_validation
from sklearn.datasets.samples_generator import make_blobs

#define the function of plot
def plot_classifier(classifier, X, y):
    # define ranges to plot the figure 
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot 
    plt.figure()

    # choose a color scheme you can find all the options 
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    # Overlay the training points on the plot 
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

    plt.show()

if __name__=='__main__':
    
    ##Create data
    (X, y) = make_blobs(n_samples=400, n_features=2, centers=2, cluster_std=2.5, random_state=95)
    
    # train test split
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.20, random_state=None)
    
    # initialize the logistic regression classifier
    classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

    # train the classifier
    classifier.fit(X, y)
    
    # predict
    y_pred = classifier.predict(X)
    y_test_pred = classifier.predict(X_test)   
    
    # compute accuracy of the classifier of test data
    accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
    print ("Accuracy of the classifier of test data =", round(accuracy, 2), "%")
    
    #plot the test data
    plot_classifier(classifier, X_test, y_test)

    
    # compute accuracy of the classifier
    accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
    print ("Accuracy of the classifier =", round(accuracy, 2), "%")
    
    #plot the test data
    plot_classifier(classifier, X, y)
    
    # Cross validation and scoring functions
    accuracy = cross_validation.cross_val_score(classifier, X, y, scoring='accuracy', cv=5)
    print ("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")

    f1 = cross_validation.cross_val_score(classifier, X, y, scoring='f1_weighted', cv=5)
    print ("F1: " + str(round(100*f1.mean(), 2)) + "%")

    precision = cross_validation.cross_val_score(classifier, X, y, scoring='precision_weighted', cv=5)
    print ("Precision: " + str(round(100*precision.mean(), 2)) + "%")

    recall = cross_validation.cross_val_score(classifier, X, y, scoring='recall_weighted', cv=5)
    print ("Recall: " + str(round(100*recall.mean(), 2)) + "%")
    