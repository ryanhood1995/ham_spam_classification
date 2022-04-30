# =======================================================================================================
# Author: Ryan Hood
#
# Description: This script takes directories containing ham/spam text files, and trains different models
# base on the training sets and predicts the identity of files in the testing sets.  Discrete Naive Bayes,
# Multinomial Naive Bayes, and Logistic Regression are developed.  For more information, please see the
# READ_ME.txt file for more information.
# =======================================================================================================

# First we import the required libraries.
import os
import math
import sys
import random
import copy
import numpy

# =======================================================================================================
# First we create the methods responsible for putting the data in a canonical form.
# =======================================================================================================

def get_complete_file_list(ham_dir, spam_dir):
    """ This method goes through both directories and returns a single list of all files contained in those directories."""
    ham_list = os.listdir(ham_dir)
    spam_list = os.listdir(spam_dir)
    final_list = ham_list + spam_list
    return final_list

def separate_file_list(ham_dir, spam_dir, file_list):
    """ This method takes ham and spam directories and a file list containing file names that may exist in either directory.  The method
    separates the given file list into two lists.  The resulting ham list only contains files present in the ham directory, and similarly
    for the spam list."""
    ham_dir_list = os.listdir(ham_dir)
    spam_dir_list = os.listdir(spam_dir)
    ham_list = []
    spam_list = []
    for file in file_list:
        if file in ham_dir_list:
            ham_list.append(file)
        elif file in spam_dir_list:
            spam_list.append(file)
        else:
            print("ERROR in method separate_file_list()")
    return ham_list, spam_list

def remove_duplicates_from_vocabulary(input_list):
    """ This method takes a list and removes the duplicates from that list.  This is used instead of the set method because the
    set method disrupts and randomizes the ordering."""
    unique_words = []
    for word in input_list:
        if word not in unique_words:
            unique_words.append(word)
    return unique_words

def get_vocabulary(ham_dir, spam_dir, file_list):
    """ This method takes a ham and spam directory along with a list of files.  For all files included in the file list, this method
    returns a list containing all of the unique words.  The resulting list is the vocabulary we will use."""
    unique_words = []
    ham_list, spam_list = separate_file_list(ham_dir, spam_dir, file_list)
    # First we go through the ham directory.
    for file_name in ham_list:
        file = open(ham_dir + file_name, "r", encoding="utf8", errors='ignore')
        initial_string = file.read().lower()
        new_list = initial_string.split()
        unique_words = unique_words + new_list
        file.close()
    # Now we go through the spam_directory.
    for file_name in spam_list:
        file = open(spam_dir + file_name, "r", encoding="utf8", errors='ignore')
        initial_string = file.read().lower()
        new_list = initial_string.split()
        unique_words = unique_words + new_list
        file.close()
    # Now unique_words contains all words in the directory of text files.  We need to make them truly unique.
    unique_words = remove_duplicates_from_vocabulary(unique_words)
    return unique_words

def get_canonical_representation(vocabulary, ham_dir, spam_dir, file_list, canonical_choice):
    """ This method takes two directories (a ham and a spam), and returns the canonical data representation.  Specifically,
    a list of lists is returned, each sub-list corresponds to a file.  Thus if the dictionary is w words long, and
    the number of files is f, then the resulting list will contain f sub-lists each with w elements.
    The sub-lists start with the ham lists and then the spam lists."""
    final_list = []
    ham_list, spam_list = separate_file_list(ham_dir, spam_dir, file_list)
    for file_name in ham_list:
        file = open(ham_dir + file_name, "r", encoding="utf8", errors='ignore')
        file_string = file.read().lower()
        word_list = file_string.split()
        # Initialize to a list of zeros.
        word_counts = [0]*len(vocabulary)
        for word in word_list:
            # First find index of the word in the vocabulary.
            word_in_vocabulary = word in vocabulary
            if not word_in_vocabulary:
                continue
            vocab_index = vocabulary.index(word)
            # Set the value to 1 or add 1 depending on the canonical representation desired.
            if canonical_choice == 0:
                word_counts[vocab_index] = 1
            else:
                word_counts[vocab_index] = word_counts[vocab_index] + 1
        # Now for the current file, word_counts has been finalized.  So we append word_counts
        # to the final list.
        final_list.append(word_counts)
    # At this point, all files in the ham_list have been completed.  So we do the same for the spam_list.
    for file_name in spam_list:
        file = open(spam_dir + file_name, "r", encoding="utf8", errors='ignore')
        file_string = file.read().lower()
        word_list = file_string.split()
        word_counts = [0]*len(vocabulary)
        for word in word_list:
            word_in_vocabulary = word in vocabulary
            if not word_in_vocabulary:
                continue
            vocab_index = vocabulary.index(word)
            if canonical_choice == 0:
                word_counts[vocab_index] = 1
            else:
                word_counts[vocab_index] = word_counts[vocab_index] + 1
        final_list.append(word_counts)
    return final_list


def get_correct_list(ham_test_dir, spam_test_dir):
    """ This method takes two desting directories, and returns the a list of the true identities of all files within.
    The resulting list starts with the identity of the ham files, and then the spam."""
    return [0]*len(os.listdir(ham_test_dir)) + [1]*len(os.listdir(spam_test_dir))


# =======================================================================================================
# Below are the methods responsible for the Multinomial Naive Bayes.
# =======================================================================================================
def multinomial_bayes_predictions(ham_train_dir, spam_train_dir, ham_test_dir, spam_test_dir):
    """ This method takes a directory and gets the class prediction for each file in that directory.
    The result is a list with all of the predictions. """
    train_file_list = get_complete_file_list(ham_train_dir, spam_train_dir)

    # First let's create the vocabularies.
    vocabulary = get_vocabulary(ham_train_dir, spam_train_dir, train_file_list)
    canonical_representation = get_canonical_representation(vocabulary, ham_train_dir, spam_train_dir, train_file_list, 1)

    # Now that those have been initialized, we can get the predictions for every file in test_directory.
    prediction_list = []
    test_file_list = get_complete_file_list(ham_test_dir, spam_test_dir)
    ham_test_list, spam_test_list = separate_file_list(ham_test_dir, spam_test_dir, test_file_list)

    for file_name in ham_test_list:
        print("File name: ", file_name)
        file = open(ham_test_dir + file_name, "r", encoding="utf8", errors='ignore')
        prediction = multinomial_bayes_file_prediction(file, ham_train_dir, spam_train_dir, canonical_representation, vocabulary)
        prediction_list.append(prediction)
        print("File prediction: ", prediction)
        print("==========================================================")
    for file_name in spam_test_list:
        print("File name: ", file_name)
        file = open(spam_test_dir + file_name, "r", encoding="utf8", errors='ignore')
        prediction = multinomial_bayes_file_prediction(file, ham_train_dir, spam_train_dir, canonical_representation, vocabulary)
        prediction_list.append(prediction)
        print("File prediction: ", prediction)
        print("==========================================================")
    return prediction_list

def multinomial_bayes_file_prediction(file, ham_train_dir, spam_train_dir, canonical_representation, vocabulary):
    """ This method takes a file and returns the prediction (0 for ham, 1 for spam) for the particular file."""

    # The first step is to generate the file_words list.
    words = file_words(file)

    # Now let's get the prior probabilities.
    ham_prior = get_prior_probabilities(ham_train_dir, spam_train_dir)

    ham_size = len(os.listdir(ham_train_dir))

    # Now let's explicitly get the log liklihood for class 0 (ham) of the file.
    # We first get the total number of occurences of the vocabulary in the ham_bag.
    ham_total_occurences = total_occurences_in_representation(canonical_representation, ham_size, 0)
    spam_total_occurences = total_occurences_in_representation(canonical_representation, ham_size, 1)

    # Now we do 1-laplace smoothing for the above term by adding the size of the vocabulary.
    ham_total_occurences = ham_total_occurences + len(vocabulary)
    spam_total_occurences = spam_total_occurences + len(vocabulary)

    ham_score = math.log(ham_prior)
    for word in words:
        # We get the number of occurences of the word in canonical_representation.  The +1 is laplace smoothing.
        word_occurences = word_occurences_in_representation(word, canonical_representation, vocabulary, ham_size, 0) + 1
        ham_score = ham_score + math.log(word_occurences / ham_total_occurences)

    # Now let's explicitly get the log liklihood for class 1 (spam) of the file.

    spam_score = math.log(1-ham_prior)
    for word in words:
        word_occurences = word_occurences_in_representation(word, canonical_representation, vocabulary, ham_size, 1) + 1
        spam_score = spam_score + math.log(word_occurences / spam_total_occurences)

    print("Ham score: ", ham_score)
    print("Spam score:  ", ham_score)

    # Now we compare the two likelihoods and return 0 if ham is higher, or 1 if spam is higher.
    if ham_score > spam_score:
        return 0
    elif spam_score > ham_score:
        return 1
    else:
        return -1


def get_prior_probabilities(ham_directory, spam_directory):
    """ This method takes a ham and spam directory, and returns the prior probabilities of the a new file being
    ham or spam. A single number is returned which represents P(ham).  P(ham) + P(spam) = 1."""
    prior_probabilities = []
    num_files_ham = len(next(os.walk(ham_directory))[2])
    num_files_spam = len(next(os.walk(spam_directory))[2])
    total_files = num_files_ham + num_files_spam
    return (num_files_ham / total_files)

def word_occurences_in_representation(word, canonical_representation, vocabulary, ham_size, choice):
    """ This method takes a word in the current file under consideration, finds the vocab index of the word
    and then counts the total number of times that word is in the bag-of-words representation.
    The choice dictates which part of the canonical representation is used.  If choice == 0,
    we use the top part (ham).  If choice == 1, we use the bottom part (spam). """
    # First, we find the index of word in vocabulary.
    # First check if the item is in the list
    word_in_vocabulary = word in vocabulary
    if not word_in_vocabulary:
        return 0
    # If not, get its index.
    index = vocabulary.index(word)

    # Now index contains the index of word in vocabulary.
    # Before we start counting, let's initialize a count.
    count = 0
    # Now we go through each list in bag-of-words and add the value at the index to our running count.
    if choice == 0:
        for row_index in range(0, ham_size):
            row = canonical_representation[row_index]
            row_count = row[index]
            count = count + row_count
    elif choice == 1:
        for row_index in range(ham_size, len(canonical_representation)):
            row = canonical_representation[row_index]
            row_count = row[index]
            count = count + row_count
    # At this point, count contains the number of times the particular word occured in the bag-representation.
    return count

def file_occurences_in_representation(file_words, canonical_representation, vocabulary):
    """ This method takes a list of words in a file (w/ repeats) and find the total number of times
    all of the words in the file show up in the bag-of-words representation."""
    # We first initialize a total_count.
    total_count = 0
    # For each word in the file_words list, we find the occurences, and add them to the running count.
    for word in file_words:
        total_count = total_count + word_occurences_in_representation(word, canonical_representation, vocabulary)
    return total_count

def total_occurences_in_representation(representation, ham_size, choice):
    """ This method finds the total number of occurences in a section of the canonical representation.  The value of
    choice dictates which section of the canonical representation is searched.  If choice = 0, then the ham part is searched.
    If choice = 1, then the spam part is searched."""
    count = 0
    if choice == 0:
        for row_index in range(0, ham_size):
            row = representation[row_index]
            for number in row:
                count = count + number
    elif choice == 1:
        for row_index in range(ham_size, len(representation)):
            row = representation[row_index]
            for number in row:
                count = count + number
    else:
        print("ERROR in method total_occurences_in_representation()")
    return count


def file_words(file):
    """ This method takes a file and returns a list of all words (w/ repeats) in the file"""
    file_string = file.read().lower()
    word_list = file_string.split()
    return word_list


# =======================================================================================================
# Below are the methods responsible for the Discrete Naive Bayes.
# =======================================================================================================

def discrete_bayes_predictions(ham_train_dir, spam_train_dir, ham_test_dir, spam_test_dir):
    """ This method takes the 4 relevant directories and returns a list of predictions for every file in the
    test directories."""
    # First let's create the vocabulary.
    total_file_list = get_complete_file_list(ham_train_dir, spam_train_dir)
    vocabulary = get_vocabulary(ham_train_dir, spam_train_dir, total_file_list)

    # Now let's get the canonical representations.
    canonical_representation = get_canonical_representation(vocabulary, ham_train_dir, spam_train_dir, total_file_list, 0)
    prediction_list = []
    for file_name in os.listdir(ham_test_dir):
        print("File name: ", file_name)
        file = open(ham_test_dir + file_name, "r", encoding="utf8", errors='ignore')
        prediction = discrete_bayes_file_prediction(file, ham_train_dir, spam_train_dir, canonical_representation, vocabulary)
        prediction_list.append(prediction)
        print("File prediction: ", prediction)
        print("==========================================================")
    for file_name in os.listdir(spam_test_dir):
        print("File name: ", file_name)
        file = open(spam_test_dir + file_name, "r", encoding="utf8", errors='ignore')
        prediction = discrete_bayes_file_prediction(file, ham_train_dir, spam_train_dir, canonical_representation, vocabulary)
        prediction_list.append(prediction)
        print("File prediction: ", prediction)
        print("==========================================================")
    return prediction_list

def discrete_bayes_file_prediction(file, ham_train_dir, spam_train_dir, canonical_representation, vocabulary):
    """ This method takes a single file and returns a prediction (0 or 1) for that file."""
    # First let's get a list of the words in the file.
    words = file_words(file)

    # Let's initialize our ham and spam scores using the prior probabilities.
    ham_prior = get_prior_probabilities(ham_train_dir, spam_train_dir)
    ham_score = math.log(ham_prior)
    spam_score = math.log(1-ham_prior)

    # Now let's get the size of both directories.
    ham_directory_size = len(next(os.walk(ham_train_dir))[2])
    spam_directory_size = len(next(os.walk(spam_train_dir))[2])


    # Now let's loop through the words in our words list, updating our scores at each step.
    for word in words:

        num_occurences_ham = word_occurences_in_representation(word, canonical_representation, vocabulary, ham_directory_size, 0)
        num_occurences_spam = word_occurences_in_representation(word, canonical_representation, vocabulary, ham_directory_size, 1)

        # The below lines may need to be changed for 1-Laplace smoothing.  instead of +2, add the number of distinct words in directory
        map_estimate_ham = (num_occurences_ham + 1)/(ham_directory_size + 2)
        map_estimate_spam = (num_occurences_spam + 1)/(spam_directory_size + 2)

        # Now update the ham and spam scores.
        ham_score = ham_score + math.log(map_estimate_ham)
        spam_score = spam_score + math.log(map_estimate_spam)

    # Now we have looped through all words and our ham and spam scores are set.  Let's make a prediction.
    if ham_score > spam_score:
        return 0 # Predict Ham.
    elif ham_score < spam_score:
        return 1 # Predict Spam.
    else:
        return -1 # Very Rare Event: no prediction is made.



# =======================================================================================================
# Below are the methods responsible for the Logistic Regression.
# =======================================================================================================

def train_test_split(ham_train_dir, spam_train_dir):
    """ This method performs a 70-30 split of the training data.  It returns 4 lists, each containing
    the names of the files that are now in that set."""
    num_files_ham = len(next(os.walk(ham_train_dir))[2])
    num_files_spam = len(next(os.walk(spam_train_dir))[2])
    num_val_ham = int(math.ceil(0.3*num_files_ham))
    num_val_spam = int(math.ceil(0.3*num_files_spam))

    # files_list contains every file in both directories.
    files_list_ham = os.listdir(ham_train_dir)
    files_list_spam = os.listdir(spam_train_dir)
    files_list = files_list_ham + files_list_spam

    val_set_ham = random.sample(files_list_ham, num_val_ham)
    val_set_spam = random.sample(files_list_spam, num_val_spam)
    train_set_ham = [x for x in files_list_ham if x not in val_set_ham]
    train_set_spam = [x for x in files_list_spam if x not in val_set_spam]
    return train_set_ham, train_set_spam, val_set_ham, val_set_spam

def get_combined_canonical_representation_from_files(vocabulary, ham_train_dir, spam_train_dir, ham_train_list, spam_train_list, canonical_choice):
    """ This method returns a canonical representation of data given the ham and spam directories and the list
    of files that are desired in that canonical representation. """
    # We first combine the two given file list into one.
    total_file_list = ham_train_list + spam_train_list
    # Call previous function to get representation.
    canonical_representation = get_canonical_representation(vocabulary, ham_train_dir, spam_train_dir, total_file_list, canonical_choice)
    return canonical_representation

def gradient_descent(canonical_representation, class_values, learning_rate, lambda_val):
    """ This method performs gradient descent and returns an optimal weight vector. """
    # First we initialize the weight vector to all zeros.
    weights = [0]*len(canonical_representation[0])

    # Then we set the number of iterations of gradient descent to do.
    num_iterations = 25

    for descent_iteration in range(0, num_iterations): # We perform this many iterations of gradient descent.
        print("Gradient Descent Iteration: ", descent_iteration, " / ", num_iterations)

        # We pre-compute a list of P(Yl = 1 | Xl, w) to use later.
        pre_computed_probabilities = [0]*len(weights)
        for l in range(0, len(canonical_representation)):
            Xi_list = canonical_representation[l]
            prob_term = calculate_exp_term(weights, Xi_list)
            pre_computed_probabilities[l] = prob_term


        new_weight_list = [0]*len(weights)


        for weight_index in range(0, len(weights)): # This for loop goes over every weight.
            # We initialize a new_weight value to the starting weight value and the sum value to 0.
            new_weight = weights[weight_index]
            sum = 0

            for sum_index in range(0, len(canonical_representation)): # This for loop calculates the sum to adjust each weight.
                # First get Xil and Xi_list.
                Xil = canonical_representation[sum_index][weight_index] # This is a single value.
                Xi_list = canonical_representation[sum_index] # This is a list.
                error = class_values[sum_index] - pre_computed_probabilities[sum_index]
                sum = sum + (Xil)*error

            # Now the sum has been set.  We adjust the weight value
            new_weight = new_weight + (learning_rate)*sum - (learning_rate)*(lambda_val)*new_weight

            # Now that the weight has been set, we reassign and move on.
            new_weight_list[weight_index] = new_weight

        # Now all new weights are in the new_weight_list.
        weights = copy.copy(new_weight_list)

    return weights

def calculate_exp_term(weight_vector, Xi_list):
    """ This method calls calculate_sum_term, then calculates exp(s)/(1+exp(s))."""
    sum_term = calculate_sum_term(weight_vector, Xi_list)
    return (1/(1+numpy.exp(-sum_term)))

def calculate_sum_term(weight_vector, Xi_list):
    """ This method takes two list, and essentially returns the dot product of the two lists."""
    # We initialize sum to be w0.
    sum = weight_vector[0]
    for index in range(1, len(weight_vector)):
        sum = sum + weight_vector[index]*Xi_list[index]
    return sum

def get_predictions(weight_vector, Xi_list):
    prob_1 = calculate_exp_term(weight_vector, Xi_list)
    if prob_1 > 0.50:
        return 1
    elif prob_1 < 0.50:
        return 0
    else:
        return -1


def classify_logistic_regression(canonical_representation, class_values, test_representation, learning_rate, lambda_val):
    optimized_weight_vector = gradient_descent(canonical_representation, class_values, learning_rate, lambda_val)
    predictions = []
    for row in test_representation:
        prediction = get_predictions(optimized_weight_vector, row)
        predictions.append(prediction)
    return predictions

def get_class_values(ham_dir, spam_dir, total_files):
    ham_list, spam_list = separate_file_list(ham_dir, spam_dir, total_files)
    return [0]*len(ham_list) + [1]*len(spam_list)


def logistic_regression_driver(ham_train_dir, spam_train_dir, ham_test_dir, spam_test_dir, canonical_choice):
    """ This method performs logistic regression against a validation set to select the best value of lambda.
    Then, a model with the best value for lambda is used against a test set."""

    # First, we perform the 70-30 split.
    train_ham, train_spam, val_ham, val_spam = train_test_split(ham_train_dir, spam_train_dir) # The results are lists of files.
    # and form the total file lists.
    train_total_files = train_ham + train_spam
    val_total_files = val_ham + val_spam

    # Get the vocbulary.
    vocabulary = get_vocabulary(ham_train_dir, spam_train_dir, train_total_files)

    # Then we get the canonical representations.
    train_representation = get_canonical_representation(vocabulary, ham_train_dir, spam_train_dir, train_total_files, canonical_choice)
    val_representation = get_canonical_representation(vocabulary, ham_train_dir, spam_train_dir, val_total_files, canonical_choice)

    # We need the class values for both the train and val representations.
    train_class_values = get_class_values(ham_train_dir, spam_train_dir, train_total_files)
    val_class_values = get_class_values(ham_train_dir, spam_train_dir, val_total_files)

    # Once we have the canonical representations, we perform logistic regression to determine best lambda value.
    possible_lambda_values = [0.01, 0.1, 0.5, 1, 5, 20]
    best_lambda_value = -1
    best_accuracy = 0.0
    for lambda_val in possible_lambda_values:
        print("Currently constructing model for lambda = ", lambda_val)
        predictions = classify_logistic_regression(train_representation, train_class_values, val_representation, 0.1, lambda_val)

        # Get accuracy between real values (from val set) and predictions list
        accuracy, precision, recall, f1_score = get_performance_metrics(val_class_values, predictions)
        print("Accuracy for lambda = ", lambda_val, " is ", accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lambda_value = lambda_val

    # At this point, best_lambda_value is set, so we apply it to the full training set.
    total_file_list = get_complete_file_list(ham_train_dir, spam_train_dir)
    combined_canonical_representation = get_canonical_representation(vocabulary, ham_train_dir, spam_train_dir, total_file_list, canonical_choice)
    combined_class_values = get_class_values(ham_train_dir, spam_train_dir, total_file_list)

    test_file_list = get_complete_file_list(ham_test_dir, spam_test_dir)
    test_representation = get_canonical_representation(vocabulary, ham_test_dir, spam_test_dir, test_file_list, canonical_choice)

    # Once we have the canonical representation, we perform logistic regression with our best lambda.
    predictions = classify_logistic_regression(combined_canonical_representation, combined_class_values, test_representation, 0.1, best_lambda_value)


    print("The best value for lambda is: ", best_lambda_value)
    return predictions


# =======================================================================================================
# Below are methods relating to measuring the performance.
# =======================================================================================================

def build_confusion_matrix(prediction_list, correct_list):
    true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
    positive_indices, negative_indices = [], []

    # First we get the indices that correspond to positive and negative predictions.
    for index in range(0, len(prediction_list)):
        if prediction_list[index] == 1:
            positive_indices.append(index)
        else:
            negative_indices.append(index)

    # Now we get each element of the confusion matrix.
    for index in positive_indices:
        if correct_list[index] == 1:
            true_positives = true_positives + 1
        else:
            false_positives = false_positives + 1
    for index in negative_indices:
        if correct_list[index] == 1:
            false_negatives = false_negatives + 1
        else:
            true_negatives = true_negatives + 1

    return true_positives, false_positives, true_negatives, false_negatives

def get_performance_metrics(prediction_list, correct_list):
    true_positives, false_positives, true_negatives, false_negatives = build_confusion_matrix(prediction_list, correct_list)
    accuracy = ((true_positives + true_negatives)/(true_positives + false_positives + true_negatives + false_negatives))
    if (true_positives + false_positives) == 0:
        precision = "N/A"
    else:
        precision = ((true_positives)/(true_positives + false_positives))
    if (true_positives + false_negatives == 0):
        recall = "N/A"
    else:
        recall = ((true_positives)/(true_positives + false_negatives))
    if (precision == "N/A" or recall == "N/A"):
        f1_score = "N/A"
    else:
        f1_score = f1_score = (2 * recall * precision)/(recall + precision)
    return accuracy, precision, recall, f1_score

# =======================================================================================================
# Below is the main, which drives the whole program.  It is presented in menu format.
# =======================================================================================================

if __name__ == '__main__':
    # Set default arguments:
    ham_train_dir = "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\train_1\\ham\\"
    spam_train_dir = "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\train_1\\spam\\"
    ham_test_dir = "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\test_1\\ham\\"
    spam_test_dir = "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\test_1\\spam\\"
    algorithm_choice = 1

    # Set new arguments if the correct number of them were given.
    if len(sys.argv) == 6:
        print("You provided the correct number of parameters.  Congrats!")
        ham_train_dir = sys.argv[1]
        spam_train_dir = sys.argv[2]
        ham_test_dir = sys.argv[3]
        spam_test_dir = sys.argv[4]
        algorithm_choice = int(sys.argv[5])
    else:
        print("You did not provide the correct number of parameters.  Using default selections.")

    if algorithm_choice == 0:
        print("Please give me ~ 2 minutes while I process the Multinomial Bayes model.")
        prediction_list = multinomial_bayes_predictions(ham_train_dir, spam_train_dir, ham_test_dir, spam_test_dir)
        correct_list = get_correct_list(ham_test_dir, spam_test_dir)
        accuracy, precision, recall, f1_score = get_performance_metrics(prediction_list, correct_list)
    elif algorithm_choice == 1:
        print("Please give me ~ 30 seconds minutes while I process the Discrete Bayes model.")
        prediction_list = discrete_bayes_predictions(ham_train_dir, spam_train_dir, ham_test_dir, spam_test_dir)
        correct_list = get_correct_list(ham_test_dir, spam_test_dir)
        accuracy, precision, recall, f1_score = get_performance_metrics(prediction_list, correct_list)
    elif algorithm_choice == 2:
        print("Please give me some time while I process the Logistic Regression Bag of Words model.")
        prediction_list = logistic_regression_driver(ham_train_dir, spam_train_dir, ham_test_dir, spam_test_dir, algorithm_choice)
        correct_list = get_correct_list(ham_test_dir, spam_test_dir)
        accuracy, precision, recall, f1_score = get_performance_metrics(prediction_list, correct_list)
    elif algorithm_choice == 3:
        print("Please give me some time while I process the Logistic Regression Bernoulli model.")
        prediction_list = logistic_regression_driver(ham_train_dir, spam_train_dir, ham_test_dir, spam_test_dir, algorithm_choice)
        correct_list = get_correct_list(ham_test_dir, spam_test_dir)
        accuracy, precision, recall, f1_score = get_performance_metrics(prediction_list, correct_list)
    else:
        print("An invalid argument for the algorithm choice was used.  It must be a whole number from 0 to 7 inclusive.")

    print("====================================================================================================================================")
    print("These are the predictions: ", prediction_list)
    print("====================================================================================================================================")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score)
