import numpy as np

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.
    
    Parámetros
    ------------
    eta : float
        Tasa aprendizaje (entre 0.0 y 1.0)
    n_iter : int
        Pases sobre los datos de entrenamiento.
    shuffle : bool (default: True)
        Baraha los datos de entrenamiento cada epoca si True
        para impedir ciclos.
    random_state : int
        Semilla generadora de numeros aleatorios para la 
        inicialización de pesos aleatoria.
    
    
    Atributos
    ---------
    w_ : 1d-array
        Pesos despues del ajuste.
    cost_ : list
        Valor de la funcion de coste suma de cuadrados promediado
        sobre todos los ejemplos de entrenamiento en cada época.
    
    """
    def __init__(self, eta=0.01, n_iter=10,
              shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        """ Ajuste de los datos de entrenamiento.
        
        Parámetros
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of 
            examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.
        
        Returns
        -------
        self : object
        
        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self
    
    def _shuffle(self, X, y):
        """Baraja los ejemplos de entrenamiento"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Inicializa los pesos a números aleatorios pequeños"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01,
                                   size=1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Aplica la regla de aprendizaje Adaline para acutalizar
        los pesos
        
        """
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class laberl after unit step"""
        return np.where(self.activation(self.net_input(X))
                        >= 0.0, 1, -1)