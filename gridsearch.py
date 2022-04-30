# =================================================================================================================================================================================
# Author: Ryan Hood
#
# Description: This python file performs GridSearch using a Stochastic Gradient Descent Classifier, and the uses the best model to predict
# ham/spam messages.  Please see the READ_ME.txt file for more information.
# =================================================================================================================================================================================


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import hw2_main
import numpy
import os
import sys

# The only additional methods needed are those which construct the representations.
def create_X_and_y_train(vocabulary, ham_train_dir, spam_train_dir, canonical_choice):
    """ This method creates a canonical representation for the training files for the desired representation."""
    total_file_list = hw2_main.get_complete_file_list(ham_train_dir, spam_train_dir)
    canonical_representation = hw2_main.get_canonical_representation(vocabulary, ham_train_dir, spam_train_dir, total_file_list, canonical_choice)

    ham_dir_length = len(os.listdir(ham_train_dir))
    spam_dir_length = len(os.listdir(spam_train_dir))


    zero_list = [0]*ham_dir_length
    one_list = [1]*spam_dir_length

    y_list = zero_list + one_list

    final_representation = numpy.array(canonical_representation)
    final_y_list = numpy.array(y_list)

    return final_representation, final_y_list

def create_X_and_y_test(vocabulary, ham_test_dir, spam_test_dir, canonical_choice):
    """ This method creates the canonical representation for the testing files for the desired representation.1 """
    total_file_list = hw2_main.get_complete_file_list(ham_test_dir, spam_test_dir)
    canonical_representation = hw2_main.get_canonical_representation(vocabulary, ham_test_dir, spam_test_dir, total_file_list, canonical_choice)

    ham_dir_length = len(os.listdir(ham_test_dir))
    spam_dir_length = len(os.listdir(spam_test_dir))

    zero_list = [0]*ham_dir_length
    one_list = [1]*spam_dir_length

    y_list = zero_list + one_list

    final_representation = numpy.array(canonical_representation)
    final_y_list = numpy.array(y_list)

    return final_representation, final_y_list



if __name__ == '__main__':
    # Set default arguments.
    ham_train_dir = "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\train_3\\ham\\"
    spam_train_dir = "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\train_3\\spam\\"
    ham_test_dir = "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\test_3\\ham\\"
    spam_test_dir = "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\test_3\\spam\\"
    canonical_representation_choice = 1

    # Set new arguments if the correct number of them were given.
    if len(sys.argv) == 6:
        print("You provided the correct number of parameters.  Congrats!")
        ham_train_dir = sys.argv[1]
        spam_train_dir = sys.argv[2]
        ham_test_dir = sys.argv[3]
        spam_test_dir = sys.argv[4]
        canonical_representation_choice = int(sys.argv[5])
    else:
        print("You did not provide the correct number of parameters.  Using default selections.")

    # Perform the Process.
    if canonical_representation_choice == 0:
        print("Please give me ~ 2 minutes to do my calculations.")
    else:
        print("Please give me ~ 4 minutes to do my calculations.")

    # First get the vocabulary and the various representations.
    total_file_list = hw2_main.get_complete_file_list(ham_train_dir, spam_train_dir)
    vocabulary = hw2_main.get_vocabulary(ham_train_dir, spam_train_dir, total_file_list)

    X_train, y_train = create_X_and_y_train(vocabulary, ham_train_dir, spam_train_dir, canonical_representation_choice)
    X_test, y_test = create_X_and_y_test(vocabulary, ham_test_dir, spam_test_dir, canonical_representation_choice)

    # Now we create the SGD Classifier.
    sgd = SGDClassifier(max_iter = 5000)

    # Below I set up my possibilities for the grid.
    loss_options = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
    penalty_options = ['l2', 'l1', 'elasticnet']
    fit_intercept_options = [True, False]

    param_grid = dict(loss = loss_options, penalty = penalty_options, fit_intercept = fit_intercept_options)
    grid = GridSearchCV(sgd, param_grid, cv=10, scoring='accuracy')

    grid.fit(X_train,y_train)

    # Print the best model parameters from grid search.
    print("The best parameters values are: ", grid.best_params_)

    # Now we train a model using those parameters.
    print("Creating model with the best parameters.")
    best_model = grid.best_estimator_

    print("Model Created.")
    print("Classifying Test Set.")
    predictions = best_model.predict(X_test)
    print("These are the predictions for the test set: ", predictions)

    prediction_list = predictions.tolist()
    correct_list = y_test.tolist()
    accuracy, precision, recall, f1_score = hw2_main.get_performance_metrics(prediction_list, correct_list)
    print("Below are the scores for the best model.")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score)
