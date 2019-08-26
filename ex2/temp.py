for _ in range(n_iter):   
    output1, output2 = self.forward_pass(x_train)
    
    # Error and derivate of 2nd layer
    layer2_error = (y_train - output2)
    if not self.classification:
        layer2_error = (layer2_error).sum(axis=1)
        layer2_error = layer2_error.reshape(-1, 1)
    layer2_delta = layer2_error * (output2*(1-output2))

    # Error and derivate of 1st layer
    #layer1_error = layer2_delta.dot(self.weights[1].T)
    layer1_error = np.dot(layer2_delta, self.weights[1].T)
    layer1_delta = layer1_error * output1*(1-output1)

    # Update weights with learning rate
    self.grad1 = x_train.T.dot(layer1_delta)
    self.grad2 = output1.T.dot(layer2_delta)
    if verbose:
        print("Total loss:", layer2_error)
        
    self.weights[0] += self.lr*self.grad1
    self.weights[1] += self.lr*self.grad2
