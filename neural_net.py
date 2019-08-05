'''
    author : Sachin Kumar
    roll no : 108118083
    domain : Signal Processing and ML
    sub-domain : machine learning
'''

import numpy as np
from lstm import LSTM_layer
from adam_optimizer import Adam
import pandas as pd

"""

    ************************  Input to the Neural Network  **************************
        
    X - with shape (m, T_x)
        where, m  = batch size, helps us to compute many 
                        examples at the same time
                T_x = size of the sentence, shorter sentences are padded 

    ************************  NEURAL NETWORK ARCHITECTURE  ***************************

    Our Neural network would have 3 layers, basically, 

    1. Embedding layer
        Input :
            This layer would taken in the sentences, with words mapped to numbers/indexes
            Input shape would be (m, T_x)
        Output:
            Would convert the words in sentences to thier corresponding embedding vectors
            Ouput shape would be (n_x, m, T_x)
            where n_x is dimension of embedding vector
         
    2. LSTM layer
        Number of units : lstm_units, passed in when neural network is initialised
        Input : 
            A batch of sentences, with shape (n_x, BATCH_SIZE, T_x)
        Output:
            Output would be of shape (n_y, BATCH_SIZE, 1)
            1 because, it is a binary classification problem and only the last ouptut from
            the last cell in the LSTM layer is used.
    
    3. Dense layer
        Number of units : 1
        Activation : sigmoid
        Input :
            Output from the LSTM layer, would be input to this layer
            shape would be (n_y, BATCH_SIZE, 1)
        Output:
            Output shape would be (BATCH_SIZE, 1)

"""
class NeuralNetwork:
    
    
    def __init__(self, embedding_matrix, lstm_units, n_x, T_x):
        
        self.LSTM_UNITS = lstm_units
        self.EMBEDDING_MATRIX  = embedding_matrix
        self.LSTM_layer = LSTM_layer(lstm_units,  n_x, T_x)
        
        self.n_x = n_x
        #and the optimizer
        self.ADAM_OPTIMIZER = Adam(n_x, lstm_units)

        #vocab size
        self.vocab_size = self.EMBEDDING_MATRIX.shape[0]

    def initialise_parameters(self):
        
        #Initialise parameter for the LSTM layer
        lstm_params = self.LSTM_layer.initialise_params()
        
        #Initialise parameter for the single neuron
        # W1 and b1 are taken care by LSTM layer
        W2 = np.random.randn(1, self.LSTM_UNITS)
        b2 = np.zeros((1,1))
        #combine the dictionaries
        parameters = {**lstm_params, "W2" : W2, "b2" : b2}
        return parameters
    
    def train(self, X, Y, learning_rate, epochs, subset, batch_size = 64,):
        
        print('Shape of training input', X.shape)
        costs, avg_costs = [], []
        #calculate number of iterations
        num_examples = X.shape[0]

        parameters = self.initialise_parameters()  #let's go
        
        for epoch in range(epochs):
            
            #choosing random examples of size = subset, from training set
            indices = np.random.choice(num_examples, size = subset)
            X_e, Y_e = X[indices], Y[:,indices]
            iterations = int(subset/batch_size)

            for i in range(iterations):
                
                start_example, end_example = i*batch_size, (i+1)*batch_size   
                X_i = X_e[start_example:end_example, :]  #of shape (m, T_x)
                Y_i = Y_e[:, start_example:end_example]  #of shape (1, m)

                #transform the sentences into corresponding embedding vectors
                X_vec = self.Embedding_vectors(X_i)

                #Forward Propagation
                dense_cache, lstm_caches = self.forward_prop(X_vec, parameters)

                #Backward Propagation
                dense_grads, dA1 = self.backprop_Dense(Y_i, dense_cache)  #Dense layer
                lstm_grads = self.LSTM_layer.back_prop_through_time(dA1, lstm_caches, batch_size, parameters)   #LSTM layer
                
                #update parameters using gradients from this batch
                gradients = {**lstm_grads, **dense_grads}
                parameters = self.ADAM_OPTIMIZER.update_parameters(parameters, gradients, learning_rate = learning_rate)  #updating params using Adam algo

                #appending to cost list
                a2 = dense_cache["a2"]
                cost = self.cost_func(a2, Y_i)    
                costs.append(cost)
            
            mean_cost = np.mean(costs)
            print(f'Epoch {epoch+1} finished. \t  Loss : {mean_cost}')
            avg_costs.append(mean_cost)
        
        #saving parameters as they would be useful for testing
        self.parameters = parameters    
        
        return costs, avg_costs 

    def cost_func(self,A, Y):
        m = Y.shape[1]
        loss = - Y*np.log(A) - (1 - Y)*np.log(1-A)
        cost = (1/m)*np.sum(loss)
        return np.squeeze(cost)
    
    def predict(self, X_test):    
        '''
            input: X_test -- tweets from test dataset, mapped to integers and padded
            ouput: pred -- predictions by the neural network
        '''
        #transform the sentences into corresponding embedding vectors
        X_vec = self.Embedding_vectors(X_test)

        #Forward Propagation
        dense_cache, lstm_caches = self.forward_prop(X_vec, self.parameters)
        A2 = dense_cache["a2"]
        
        pred = np.zeros(A2.shape)
        indices = np.where(A2 > 0.5)   #indices where output > 0.5
        pred[indices] = 1              #are set to 1
        
        return pred
        
    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    def Embedding_vectors(self, X_i):
        '''
        input: X_i -- Tweet mapped to integers and padded
        output: X_vec -- The integers are replaced with vectors from Embedding matrix
        '''
        m, T_x = X_i.shape
        X_vec = np.zeros((self.n_x, m, T_x))
        
        for i in range(m):
            for t in range(T_x):
                X_vec[:, i, t] = self.EMBEDDING_MATRIX[int(X_i[i,t])]

        return X_vec


    def backprop_Dense(self, Y, cache):

        '''
            Backprops through the dense layer,
            returns gradients to be fed into LSTM layers
        '''
        A2, A1, W2 = cache['a2'], cache['a1'], cache['W2']
        m = Y.shape[1]
        
        dZ2 = A2 - Y
        dW2 = (1/m)*np.dot(dZ2, A1.T)
        db2 = (1/m)*np.sum(dZ2)
        dA1 = np.dot(W2.T, dZ2)
        
        dense_grads = {'dW2' : dW2, 'db2' : db2}
        return dense_grads, dA1
    
    def forward_prop(self, X_vec, parameters):

        W2, b2 = parameters['W2'], parameters['b2']
        a1, lstm_caches = self.LSTM_layer.lstm_forward(X_vec, parameters)
        z2 = np.dot(W2,a1) + b2
        a2 = self.sigmoid(z2)  #of shape (1,m)
        
        cache = {'a2': a2, 'a1': a1, 'W2': W2}
        return cache, lstm_caches
    
    def save_weights(self,directory):
        '''
            handy function, to save weights of the model
        '''
        #save files in drive for later use
        for key,param in self.parameters.items():
            df_param = pd.DataFrame(param)
            df_param.to_csv('{}/{}.csv'.format(directory, key))
        
        print('All files saved successfully')  
    
    def load_weights(self, directory):
        '''
            handy function, to load saved weights
        '''
        self.parameters = {}  #Initialising empty dictionary
        file_names = ["Wf.csv", "bf.csv", "Wi.csv", "bi.csv", "Wc.csv",
                      "bc.csv", "Wo.csv", "bo.csv","W2.csv", "b2.csv"]

        param_keys = ["Wf",  "bf", "Wi", "bi", "Wc", "bc", "Wo", "bo","W2", "b2"]

        for name, key in zip(file_names, param_keys):
            df_param = pd.read_csv(f'{directory}/{name}', header = None)
            param = df_param.values[1:,1:]
            self.parameters[key] = param

        print('Ready to go')
        