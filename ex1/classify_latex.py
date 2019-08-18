for _ in range(n_iter):   
    output = x_train.dot(self.weights) + self.bias
    
    # Derivate of weights
    dw = self.lr*(x_train.T * (y_train - output)).T
    dw = dw.sum(axis=0) / len(x_train)
    
    # Derivates of bias
    db = self.lr*(y_train - output).sum() / len(x_train)
                
    # Update
    self.weights = self.weights + dw           
    self.bias = self.bias + db
