import numpy as np

from cfgenerator.optimizers.loss_functions import constraint_loss


def optimize(x, cf_y, weights, biases, act_func, d_act_func, loss_func_dict, lr, steps, **kwargs):

    # We need to convert the shape on the input DataFrame to a numpy array
    x = x.T.to_numpy().reshape(x.shape[1],)
    # Here we take a copy of this vector, to avoid refference problems
    cf_vector = x.copy()

    for step in range(steps):
        # This list will take the input for each layer, BEFORE APPLIED the activation function
        dt = []

        # Here we make a feed forward operation, we take the as input the index of the layer (idx)
        # the layer weights and biases.
        for idx, l_weight, l_bias in zip(range(len(weights)), weights, biases):
            if len(dt) == 0:
                # This is the first feed forward step, where we take the input vector and multiply by the weights
                # and add the bias. WE DON'T APPLY THE ACTIVATION FUNCTION YET.
                dt.append(np.dot(l_weight.T, cf_vector) + l_bias)
            else:
                # Here is where all other layer iterate. THE FIRST STEP WE DO HERE IS TO APPLY THE ACTIVATION
                # FUNCTION FROM THE PREVIOUS LAYER.
                pv_act = act_func[idx - 1](dt[idx - 1])

                # Then, again, we add to the list the result of the output vector with the weights and add the bias.
                # THIS IS DONE AGAIN WITHOUT APPLYING THE ACTIVATION FUNCTION.
                dt.append(np.dot(l_weight.T, pv_act) + l_bias)

        # Here, we take the last output layer and apply the last activation function
        result = act_func[-1](dt[-1])
        

        # We print the loss for debugging purposes
        if step % 100 == 0:
            print(f"Loss: {loss_func_dict['d_loss'](result, cf_y.to_numpy(), **kwargs)}, step: {step}")

        # Here we use the user's selected loss (the gradient of it) to calculate it regarding the required output
        d_loss = loss_func_dict["d_loss"](result, cf_y.to_numpy(), **kwargs)

        # To calculate the gradient we need to calculate the following equation
        # gradient = d_loss(y, cf_y) * fa_n(dt_n)•w_n.T * fa_[n-1](dt_[n-1])•w_[n-1].T * ... * fa_1(dt_1)•w_1.T
        # where:
        #       fa_n - the n (last) layer activation function
        #       dt_n - the n (last) input result to the layer n, before applying the activation function
        #       w_n.T = the n (last) weight (Transposed .T) related to the layer n

        # The output of the gradient loss function is the first part of our total gradient calculation
        gradient = d_loss

        # Here, we iterate for each layer (non activated) output and differential form of the activation function
        for idx in range(len(dt)).__reversed__():
            # Here:
            #       f_[idx] = d_act_func[idx]
            #       dt_[idx] = dt[idx]
            #       w_[idx].T = weights[idx].T
            terms_mult = np.multiply(gradient, d_act_func[idx](dt[idx]))
            gradient = np.dot(terms_mult, weights[idx].T)

        # Bellow, commented code related to the additional gradient related to feature constraints
        # constraint_features_grad = (x-cf_vector)*[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # weight_features_grad = 0.001*(x - cf_vector)*[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # cf_vector = cf_vector - lr * gradient + constraint_features_grad + weight_features_grad

        cf_vector = cf_vector - lr * gradient

    return cf_vector