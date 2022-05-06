
import numpy as np
from numpy import linalg as LA
from scipy.linalg.blas import ddot
from scipy.linalg.blas import dscal
from scipy.linalg.blas import daxpy
from sklearn.base import BaseEstimator

class LinearClassifier(BaseEstimator):
    """
    General class for binary linear classifiers. Implements the predict
    function, which is the same for all binary linear classifiers. There are
    also two utility functions.
    """

    def decision_function(self, X):
        """
        Computes the decision function for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """
        return X.dot(self.w)

    def predict(self, X):
        """
        Predicts the outputs for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """

        # First compute the output scores
        scores = self.decision_function(X)

        # Select the positive or negative class label, depending on whether
        # the score was positive or negative.
        out = np.select([scores >= 0.0, scores < 0.0],
                        [self.positive_class,
                         self.negative_class])
        return out

    def find_classes(self, Y):
        """
        Finds the set of output classes in the output part Y of the training set.
        If there are exactly two classes, one of them is associated to positive
        classifier scores, the other one to negative scores. If the number of
        classes is not 2, an error is raised.
        """
        classes = sorted(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class = classes[1]
        self.negative_class = classes[0]

    def encode_outputs(self, Y):
        """
        A helper function that converts all outputs to +1 or -1.
        """
        return np.array([1 if y == self.positive_class else -1 for y in Y])


class Pegasos(LinearClassifier):
    """
    Implementation of the pegasos learning algorithm using hinge-loss function.
    """

    def fit(self, X, Y):
        """
        Train a linear classifier using the pegasos learning algorithm with hinge-loss.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        
        # Pegasos algorithm
        
        # Number of pairs to be randomly selected
        T = 100000
        
        # Lambda
        Lambda = 1/T
        
        for t in range(1,T+1):
            i = np.random.randint(1,len(X))
            nu = 1/(Lambda*t)

            x = X[i]
            y = Ye[i]
            score = np.dot(self.w,x)
            
            if y*score < 1:
                self.w = (1-nu*Lambda)*self.w + (nu*y)*x
            else:
                self.w = (1-nu*Lambda)*self.w
                
                
class Pegasos_BLAS(LinearClassifier):
    """
    A straightforward implementation of the pegasus learning algorithm using BLAS-functions
    """


    def fit(self, X, Y):
        """
        Train a linear classifier using the pegasos learning algorithm using BLAS-functions
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        
        # Number of pairs to be randomly selected
        T = 100000
        
        # Lambda
        Lambda = 1/T
                
        ### Pegasos algorithm using BLAS functions
        
        for t in range(1,T+1):
            i = np.random.randint(1,len(X))
            nu = 1/(Lambda*t)

            x = X[i]
            y = Ye[i]
            score = ddot(self.w,x)
            
            if y*score < 1:
                #dscal(1-nu*Lambda,self.w)
                #daxpy(x,self.w,a = ddot(nu,y))
                daxpy(x,dscal(1-ddot(nu,Lambda),self.w),a=ddot(nu,y))
                                           
            else:           
                dscal(1-nu*Lambda,self.w)
                        
        
     
                       
class Pegasos_LR(LinearClassifier):
    """
    A straightforward implementation of the pegasos learning algorithm using log-loss.
    """

    def fit(self, X, Y):
        """
        Train a linear classifier using the pegasos learning algorithm with log-loss.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        
        # Pegasos algorithm
        
        # Number of pairs to be randomly selected
        T = 100000
        epochs = np.round(T/10)
        
        # Lambda
        Lambda = 1/T
        
        sum_loss = 0
        
        for t in range(1,T+1):
            i = np.random.randint(1,len(X))
            nu = 1/(Lambda*t)

            x = X[i]
            y = Ye[i]
            score = np.dot(self.w,x)
            
            loss = -(y*x)/(1+ np.exp(y*score))
            
            sum_loss = sum_loss + np.sum(loss)
            
            self.w = self.w - (nu*loss)
            
            # printing current value of the objective function at each iteration.
            if t == epochs:
                epochs = epochs + 10000
                print(sum_loss/t + (Lambda/2)*((LA.norm(self.w))**2))
                   
                           
            
        

##### The following part is for the optional task.

### Sparse and dense vectors don't collaborate very well in NumPy/SciPy.
### Here are two utility functions that help us carry out some vector
### operations that we'll need.

def add_sparse_to_dense(x, w, factor):
    """
    Adds a sparse vector x, scaled by some factor, to a dense vector.
    This can be seen as the equivalent of w += factor * x when x is a dense
    vector.
    """
    w[x.indices] += factor * x.data

def sparse_dense_dot(x, w):
    """
    Computes the dot product between a sparse vector x and a dense vector w.
    """
    return np.dot(w[x.indices], x.data)




class SparsePegasos(LinearClassifier):
    """
    A straightforward implementation of the pegasos learning algorithm,
    assuming that the input feature matrix X is sparse.
    """


    def fit(self, X, Y):
        """
        Train a linear classifier using the pegasos learning algorithm.

        Note that this will only work if X is a sparse matrix, such as the
        output of a scikit-learn vectorizer.
        """
        self.find_classes(Y)

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        Ye = self.encode_outputs(Y)

        # Initialize the weight vector to all zeros.
        self.w = np.zeros(X.shape[1])

        # Iteration through sparse matrices can be a bit slow, so we first
        # prepare this list to speed up iteration.
        XY = list(zip(X, Ye))
        
        # number of iterations
        T = 100000
        
        # Lambda
        Lambda = 1/T
        
        for t in range(1,T+1):
            i = np.random.randint(1,len(XY))
            nu = 1/(Lambda*t)

            x = XY[i][0]
            y = XY[i][1]
           
            score = sparse_dense_dot(x,self.w)
            
            if y*score < 1:
                self.w = (1-nu*Lambda)*self.w
                add_sparse_to_dense(x,self.w,nu*y)
            else:
                self.w = (1-nu*Lambda)*self.w
                

class SparsePegasos_scale(LinearClassifier):
    """
    A straightforward implementation of the pegasos learning algorithm,
    assuming that the input feature matrix X is sparse. 
    
    In this implementation we use a scaling trick to speed up the process.
    
    """

    def __init__(self, n_iter=20):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.

        Note that this will only work if X is a sparse matrix, such as the
        output of a scikit-learn vectorizer.
        """
        self.find_classes(Y)

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        Ye = self.encode_outputs(Y)

        # Initialize the weight vector to all zeros.
        self.w = np.zeros(X.shape[1])

        # Iteration through sparse matrices can be a bit slow, so we first
        # prepare this list to speed up iteration.
        XY = list(zip(X, Ye))
        
        #print(len(XY)) 
        T = 100000
        
        # Lambda
        Lambda = 1/T
        
        a = 1 
        
        for t in range(1,T+1):
            i = np.random.randint(1,len(XY))
            nu = 1/(Lambda*t)

            x = XY[i][0]
            y = XY[i][1]
            
            a = (1-nu*Lambda)*a
           
            score = sparse_dense_dot(x,self.w)*a
            
            if y*score < 1:
                
                add_sparse_to_dense(x,self.w,(nu*y)/a)       
                
            else:
                
                self.w = self.w
          
        self.w = a*self.w
                
        

