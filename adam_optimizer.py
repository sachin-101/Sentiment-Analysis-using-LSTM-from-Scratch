import numpy as np    

class Adam():


    def __init__(self, n_x, n_a) :
        """
        Initializes v and s as two python dictionaries with:
                    - keys: "dWf", "dbf", "dWi", "dbi", "dWc", "dbc", "dWo", "dbo" , "dW2" and "db2"
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        
        Arguments:
        n_x = size of the input vectors represting each word 
        n_a = number of activation units in each LSTM cell
        """
        
        self.v = {}
        self.s = {}
        
        grad_keys =  ["dWf","dbf","dWi","dbi","dWc","dbc","dWo","dbo","dW2", "db2"]
        grad_shapes = [(n_a, n_a+n_x), (n_a,1), #shape of dWf, dbf
                       (n_a, n_a+n_x), (n_a,1), #shape of dWi, dbi
                       (n_a, n_a+n_x), (n_a,1), #shape of dWc, dbc
                       (n_a, n_a+n_x), (n_a,1), #shape of dWo, dbo
                       (1, n_a), (1,1)]         #shape of dW2, db2
        
        # Initialize dictionaries v, s 
        for key, shape in zip(grad_keys, grad_shapes):
            self.v[key] = np.zeros(shape) 
            self.s[key] = np.zeros(shape)
            

    def update_parameters(self, parameters, grads, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):   #default params are from keras docs
        """
        Update parameters using Adam
        
        Arguments:
        prameters -- python dictionary containing parameters to be updated
        grads -- python dictionary containing gradient computed using current mini-batch
        learning_rate -- the learning rate
        beta1 -- Exponential decay hyperparameter for the first moment estimates 
        beta2 -- Exponential decay hyperparameter for the second moment estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates

        Returns:
        parameters -- python dictionary containing updated parameters 
        """
     
        param_keys = list(parameters.keys())
        grad_keys = list(grads.keys())
        
        assert(grad_keys == ["dWf","dbf","dWi","dbi","dWc","dbc","dWo","dbo","dW2", "db2"])
        assert(param_keys == ["Wf",  "bf", "Wi", "bi", "Wc", "bc", "Wo", "bo","W2", "b2"])
        

        for param_key, grad_key in zip(param_keys, grad_keys):
            # Moving average of the gradients 
            self.v[grad_key] = beta1 * self.v[grad_key] + (1 - beta1) * grads[grad_key] 

            # Moving average of the squared gradients 
            self.s[grad_key] = beta2 * self.s[grad_key] + (1 - beta2) * (grads[grad_key] ** 2)

            # Update parameters
            parameters[param_key] = parameters[param_key] - learning_rate * self.v[grad_key] / np.sqrt(self.s[grad_key] + epsilon)       

        return parameters