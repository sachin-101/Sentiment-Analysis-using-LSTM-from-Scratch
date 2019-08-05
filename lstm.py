'''
    author : Sachin Kumar
    roll no : 108118083
    domain : Signal Processing and ML
    sub-domain : machine learning
'''

import numpy as np 

class LSTM_layer():

    def __init__(self, units, n_x, T_x):
        """
        Initialise the LSTM cell.

        Arguments:
        units -- positive integer, dimension of output vector
        return_sequences -- boolean, False would return only output from last cell
                            while True would return output from all cells
        
        """
        self.UNITS = units   #number of activations in the LSTM layer
        self.nx = n_x        #size of vector representing each word
        self.T_x = T_x       #size of each padded sentence

#-------------------------------Initialise parameters---------------------------------------------#


    def initialise_params(self):
        """
        Initialize parameters randomly.
        """
        n_x, n_a = self.nx, self.UNITS
        
        Wf = np.random.randn(n_a, n_a + n_x)
        bf = np.random.randn(n_a, 1)
        Wi = np.random.randn(n_a, n_a + n_x)
        bi = np.random.randn(n_a, 1)
        Wc = np.random.randn(n_a, n_a + n_x)
        bc = np.random.randn(n_a, 1)
        Wo = np.random.randn(n_a, n_a + n_x)
        bo = np.random.randn(n_a, 1)
        
        parameters = {"Wf": Wf, "bf": bf,"Wi": Wi, "bi": bi, "Wc": Wc, "bc": bc,"Wo": Wo,"bo": bo}
        return parameters

#---------------------------------Forward propagation for a single layer-------------------------------#


    def lstm_cell_forward(self, xt, a_prev, c_prev, parameters):
        """
        Implement a single forward step of the LSTM-cell as described in Figure (4)

        Arguments:
        xt -- input data at timestep "t"   
        a_prev -- Hidden state at timestep "t-1" 
        c_prev -- Memory state at timestep "t-1" 
        parameters -- dictionary containing parameters of the Layer

        Returns:
        a_next -- next hidden state
        c_next -- next memory state
        yt_pred -- prediction at timestep "t"
        cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
        
        Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
            c stands for the memory value
        """

        # Retrieve parameters from "parameters"
        Wf = parameters["Wf"]
        bf = parameters["bf"]
        Wi = parameters["Wi"]
        bi = parameters["bi"]
        Wc = parameters["Wc"]
        bc = parameters["bc"]
        Wo = parameters["Wo"]
        bo = parameters["bo"]

        # Concatenate a_prev and xt 
        concat = np.vstack((a_prev, xt))
        
        # Compute values for ft, it, cct, c_next, ot, a_next using LSTM cell formulas
        ft = sigmoid(np.dot(Wf, concat) + bf)
        it = sigmoid(np.dot(Wi, concat) + bi)
        cct = np.tanh(np.dot(Wc, concat) + bc)
        c_next = ft * c_prev + it * cct 
        ot = sigmoid(np.dot(Wo, concat) + bo)
        a_next = ot * np.tanh(c_next)
        
        # store values needed for backward propagation in cache
        cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt)

        return a_next, c_next, cache


#-----------------------------------Forward propagation for all time steps-----------------------------------------#    

    def lstm_forward(self, x, parameters):
        """
        Implement the forward propagation of the recurrent neural network using an LSTM-cell described in Figure (4).

        Arguments:
        x -- Input data for every time-step, of shape (n_x, m, T_x)
        parameters -- dictionary containing parameters of the Layer
                            
        Returns:
        a_next -- activation of the final time step to feed in the next layer
        caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
        """

        #Accepting, x.shape = n_x, m, T_x 
        #Let's make sure we are getting input of correct dimensions
        assert(self.nx == x.shape[0])
        assert(self.T_x == x.shape[2])
        
        #writing shorter variables to handle easily
        n_a, m = self.UNITS, x.shape[1]

        # Initialize "caches", which will track the list of all the caches
        caches = []

        # Initialize a_next and c_next
        a_next = np.zeros((n_a,m))   
        c_next = np.zeros(a_next.shape)
        
        # loop over all time-steps
        for t in range(self.T_x):
            # Update next hidden state, next memory state, compute the prediction, get the cache 
            a_next, c_next, cache = self.lstm_cell_forward(x[:,:,t], a_next, c_next, parameters)
            # Append the cache into caches 
            caches.append(cache)
        
        # store values needed for backward propagation in cache
        caches = (caches, x)

        #since only the last activation is of our concern
        return a_next, caches

    
#-----------------------------------------------BACKPROP for a single cell---------------------------------------#    

    def lstm_cell_backward(self, da_next, dc_next, cache, parameters):
        """
        Implement the backward pass for the LSTM-cell (single time-step).

        Arguments:
        da_next -- Gradients of next hidden state, of shape (n_a, m)
        dc_next -- Gradients of next cell state, of shape (n_a, m)
        cache -- cache storing information from the forward pass

        Returns:
        gradients -- python dictionary containing required gradients
        """
        
        # Retrieve parameters from "parameters"
        Wf = parameters["Wf"]
        Wi = parameters["Wi"]
        Wc = parameters["Wc"]
        Wo = parameters["Wo"]
        

        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt) = cache
        
        #important variables
        n_a = self.UNITS
        
        # Compute gates related derivatives
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
        dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
        dft = (dc_next * c_prev + ot *(1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)
        
        #concat them to multiply them together parameters
        concat = np.concatenate((a_prev, xt), axis=0)

        # Compute parameters related derivatives
        dWf = np.dot(dft, concat.T)
        dWi = np.dot(dit, concat.T)
        dWc = np.dot(dcct, concat.T)
        dWo = np.dot(dot, concat.T)
        dbf = np.sum(dft, axis=1 ,keepdims = True)
        dbi = np.sum(dit, axis=1, keepdims = True)
        dbc = np.sum(dcct, axis=1,  keepdims = True)
        dbo = np.sum(dot, axis=1, keepdims = True)

        # Compute derivatives w.r.t previous hidden state, previous memory state and input
        da_prev = np.dot(Wf[:, :n_a].T, dft) + np.dot(Wi[:, :n_a].T, dit) + np.dot(Wc[:, :n_a].T, dcct) + np.dot(Wo[:, :n_a].T, dot)
        dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
        dxt = np.dot(Wf[:, n_a:].T, dft) + np.dot(Wi[:, n_a:].T, dit) + np.dot(Wc[:, n_a:].T, dcct) + np.dot(Wo[:, n_a:].T, dot)
        
        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}

        return gradients



#-----------------------------Backpropagation for entire LSTM Cell, also called as Backprop through time----------------#


    def back_prop_through_time(self, da, caches, batch_size, parameters):
        
        """
        Implement the backward pass for the RNN with LSTM-cell

        Arguments:
        dy -- Gradients passed down by layers after LSTM layer (in our case the 1 unit Dense layer)
        caches -- cache storing information from the forward pass

        Returns:
        gradients -- python dictionary containing for updating parameteres
        """

        # Retrieve values from the caches
        (caches, x) = caches
        
        # Retrieve important dimensions
        n_x, n_a = self.nx, self.UNITS
        
        # initialize the gradients with the right sizes, to zeroes 
        da0 = np.zeros((n_a, batch_size))
        dc_prevt = np.zeros(da0.shape)
        dWf = np.zeros((n_a, n_a + n_x))
        dWi = np.zeros(dWf.shape)
        dWc = np.zeros(dWf.shape)
        dWo = np.zeros(dWf.shape)
        dbf = np.zeros((n_a, 1))
        dbi = np.zeros(dbf.shape)
        dbc = np.zeros(dbf.shape)
        dbo = np.zeros(dbf.shape)
        
        #from the previous layer
        da_prevt = da  

        # loop back over the whole sequence
        for t in reversed(range(self.T_x)):
            # Compute all gradients using lstm_cell_backward
            gradients = self.lstm_cell_backward(da_prevt, dc_prevt, caches[t], parameters)
            # Store or add the gradient to the parameters' previous step's gradient
            da_prevt = gradients["da_prev"]
            dc_prevt = gradients["dc_prev"]
            #sum up the gradients
            dWf += gradients["dWf"]
            dWi += gradients["dWi"]
            dWc += gradients["dWc"]
            dWo += gradients["dWo"]
            dbf += gradients["dbf"]
            dbi += gradients["dbi"]
            dbc += gradients["dbc"]
            dbo += gradients["dbo"]
            
        # Store the gradients in a python dictionary
        gradients = {"dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,"dWc": dWc,"dbc": dbc, "dWo": dWo, "dbo": dbo}

        return gradients

#---------------------------------Actiavation functions ---------------------------------------------#

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)