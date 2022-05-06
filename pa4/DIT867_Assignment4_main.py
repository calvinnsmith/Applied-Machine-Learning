{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e2d7429a",
   "metadata": {},
   "source": [
    "## DAT340/DIT867 Programming assignment 4: Implementing linear classifiers\n",
    "\n",
    "#### Calvin Smith\n",
    "#### Bragadesh Bharatwaj Sundararaman\n",
    "#### Amogha Udayakumar\n",
    "\n",
    "In order to run you need to have pegasos.py and doc_classification.py and the data in the same directory.\n",
    "\n",
    "pegasos.py contains the code for each classifier\n",
    "\n",
    "doc_classification.py runs the specified classifier\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e43eed5d",
   "metadata": {},
   "source": [
    "### Exercise question:\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5db5f0c2",
   "metadata": {},
   "source": [
    "### Question 1: Implementing the SVC Pegasos algorithm\n",
    "\n",
    "We have chosen to follow the procedure in the paper by picking a fixed number T of randomly selected pairs to iterate through. We tried a few different values, and eventually chose T = 100 000 since it produced a fairly high accuracy and at a pretty good speed. Also, choosing a number higher than 100 000 did not lead to an increase in accuracy, but only an increase in time.\n",
    "\n",
    "From this, Lambda was set to 1/T.\n",
    "\n",
    "The code for the SVC pegasos classifier can be found in pegasos.py class Pegasos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cad1e647",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training time: 3.20 sec.\n",
      "Accuracy: 0.8145.\n"
     ]
    }
   ],
   "source": [
    "## Implementing the SVC pegasos algorithm\n",
    "## In doc_classification.py use Pegasos() as classifier\n",
    "## SelectKBest should also be used\n",
    "## ngram_range is not specified\n",
    "\n",
    "from pegasos import Pegasos\n",
    "\n",
    "execfile('doc_classification.py')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b343b5ff",
   "metadata": {},
   "source": [
    "### Question 2: Implementing the LR Pegasos algorithm\n",
    "\n",
    "We use the same number of iterations T and Lmabda as in the SVC case. \n",
    "\n",
    "The code for the LR pegasos classifier can be found in pegasos.py class Pegasos_LR.\n",
    "When running the code the Pegasos_LR class will also output the value of the objective function for every 10 000 iterations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f51248b0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/calvinsmith/Documents/GitHub/DIT867/pa4/pegasos.py:211: RuntimeWarning: overflow encountered in exp\n",
      "  loss = -(y*x)/(1+ np.exp(y*score))\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.2675010157234912\n",
      "0.1043776546210023\n",
      "0.07118363907607979\n",
      "0.05743785711342846\n",
      "0.05142953244987731\n",
      "0.0481444186941109\n",
      "0.04537223340360722\n",
      "0.04399605488588447\n",
      "0.04237771038019213\n",
      "0.0417509805702854\n",
      "Training time: 5.81 sec.\n",
      "Accuracy: 0.8355.\n"
     ]
    }
   ],
   "source": [
    "## Implementing the LR pegasos algorithm\n",
    "## In doc_classification.py use Pegasos_LR() as classifier\n",
    "## SelectKBest should also be used\n",
    "## ngram_range is not specified\n",
    "\n",
    "from pegasos import Pegasos_LR\n",
    "\n",
    "execfile('doc_classification.py')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2c2679f5",
   "metadata": {},
   "source": [
    "Using the log-loss function raises the accuarcy slightly (0.8355) but with an increased training time (5.81 seconds)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "83561c75",
   "metadata": {},
   "source": [
    "### Question 3\n",
    "\n",
    "### Bonus task 1: Making your code more efficient."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "655d0c44",
   "metadata": {},
   "source": [
    "#### a) Faster linear algebra operations\n",
    "\n",
    "The code for the SVC pegasos classifier using BLAS functions can be found in pegasos.py class Pegasos_BLAS."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b7c0e5bb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training time: 2.64 sec.\n",
      "Accuracy: 0.8141.\n"
     ]
    }
   ],
   "source": [
    "## Implementing SVC pegasos algorithm using BLAS functions\n",
    "## In doc_classification.py use Pegasos_BLAS() as classifier\n",
    "## SelectKBest should also be used\n",
    "## ngram_range is not specified\n",
    "\n",
    "from pegasos import Pegasos_BLAS\n",
    "\n",
    "execfile('doc_classification.py')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c64c55f4",
   "metadata": {},
   "source": [
    "Using the BLAS function helped speed up the linear algebra operations.\n",
    "In question 1, we got a training time of 3.20 seconds and an accuracy of 0.8145. Using BLAS functions we got a training time of 2.64 seconds and \"the same\" accuracy of 0.8141.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a98953bd",
   "metadata": {},
   "source": [
    "#### b) Using sparse vectors\n",
    "\n",
    "We start by running the orgiginal SVC pegasos from question 1 but this time without using SelectKbest and changing the TFIDF vectorizer ngram range to (1,2):"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b02bac98",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training time: 453.19 sec.\n",
      "Accuracy: 0.8691.\n"
     ]
    }
   ],
   "source": [
    "from pegasos import Pegasos\n",
    "\n",
    "execfile('doc_classification.py')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3c0f906a",
   "metadata": {},
   "source": [
    "The accuracy has increased a bit, which is expected since we are utlizing a larger set fo features. However, the training time has increased significantly!\n",
    "\n",
    "Next step is to try the sparse version of SVC pegasos:\n",
    "\n",
    "The code for the sparse SVC pegasos classifier can be found in pegasos.py class SparsePegasos."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "09a6388a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training time: 126.89 sec.\n",
      "Accuracy: 0.8745.\n"
     ]
    }
   ],
   "source": [
    "## Implementing the SVC pegasos algorithm using sparse vectors\n",
    "## In doc_classification.py use SparsePegasos() as classifier\n",
    "## Remove SelectKBest \n",
    "## In the TFIDF-vectorizer, change ngram range to (1,2)\n",
    "\n",
    "from pegasos import SparsePegasos\n",
    "\n",
    "execfile('doc_classification.py')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "10e82922",
   "metadata": {},
   "source": [
    "By using sparse vectors we managed to decrease the training time from 453 seconds to 126 seconds while maintaining the accuracy."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "19b5de2d",
   "metadata": {},
   "source": [
    "#### c) Speeding up the scaling operation\n",
    "\n",
    "The code for the sparse SVC pegasos classifier with the scaling trick can be found in pegasos.py class SparsePegasos_scale."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1fa0743d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training time: 7.11 sec.\n",
      "Accuracy: 0.8691.\n"
     ]
    }
   ],
   "source": [
    "## Implementing the SVC pegasos algorithm using sparse vectors and scaling\n",
    "## In doc_classification.py use SparsePegasos_scale() as classifier\n",
    "## Remove SelectKBest \n",
    "## In the TFIDF-vectorizer, change ngram range to (1,2)\n",
    "\n",
    "\n",
    "from pegasos import SparsePegasos_scale\n",
    "\n",
    "execfile('doc_classification.py')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "95896ab0",
   "metadata": {},
   "source": [
    "With the sclaing trick the training time was dramatically reduced, from 126.89 seconds to 7.11 seconds, again the accuracy is maintained at approximately the same level (it varies because of the randomness in sampling at each iteration T)."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
