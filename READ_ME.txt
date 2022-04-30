There are two runnable files: hw2_main.py and gridsearch.py.  hw2_main.py performs Discrete Naive Bayes, 
Multinomial Naive Bayes, and Logistic Regression, and gridsearch.py performs grid search using the 
scikit library.

==============================================
How to run hw2_main.py
==============================================
hw2_main.py has 6 total arguments.  In order they are:
- the python file name
- the directory for the TRAINING HAM files
- the directory for the TRAINING SPAM files
- the directory for the TESTING HAM files
- the directory for the TESTING SPAM files
- an algorithm choice corresponding to the algorithm you wish to run

The file name is the typical convention, the directories must be provided as strings, and the algorithm choice 
must be from the set {0, 1, 2, 3}.  

- 0 corresponds to Discrete NB
- 1 corresponds to Multinomial NB
- 2 corresponds to Logistic Regression in the Bag of Words representation
- 3 corresponds to Logistic Regression in the Bernoulli representation

If the correct number of parameters is not provided, the algorithm will run with pre-set values.  The pre-set values 
can be changed on lines 569-574 of the python file.

An example command line input would be:
python hw2_main.py "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\train_1\\ham\\" "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\train_1\\spam\\" "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\test_1\\ham\\" "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\test_1\\spam\\" 1

==============================================
How to run gridsearch.py
==============================================
gridsearch.py has 6 total arguments.  In order they are:
- the python file name
- the directory for the TRAINING HAM files
- the directory for the TRAINING SPAM files
- the directory for the TESTING HAM files
- the directory for the TESTING SPAM files
- an algorithm choice corresponding to the algorithm you wish to run

The file name is the typical convention, the directories must be provided as strings, and the algorithm choice 
must be from the set {0, 1}.

- 0 corresponds to using Bernoulli representation
- 1 corresponds to using Bag-of-Words representation

If the correct number of parameters is not provided, the algorithm will run with pre-set values.  The pre-set values 
can be changed on lines 58-62 of the python file.

An example command line input would be:
python gridsearch.py "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\train_1\\ham\\" "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\train_1\\spam\\" "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\test_1\\ham\\" "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\hw2\\test_1\\spam\\" 0

Make sure to put two backslashes throughout the directory strings, and separate all arguments with a space.

-------------------------------------------------------------------------------------------------------------------------

After running either program, the best parameters, if any, the predictions of the test files, and confusion 
matrix entries should be displayed.
