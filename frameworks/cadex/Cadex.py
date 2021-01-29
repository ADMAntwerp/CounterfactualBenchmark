import keras.backend as K
import numpy as np
from keras.layers import Input, Layer
from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam
import pandas as pd

class Cadex:
    '''
    Creates counterfactual explanations based on a pre-trained model.
    The model is a Keras classifier model, where in the final classification layer, each class label must
    have a separate unit.

    '''
    def __init__(self, model, categorical_attributes=None, ordinal_attributes=None, scale=None, unscale=None):
        '''Initializes the adversarial model.
            model_file: path to a serialized keras model.
        '''
        if ordinal_attributes is not None and (scale is None or unscale is None):
            raise Exception('If ordinal attributes are specified, scale and unscale must be given')

        self.original_model = model

        self._categorical_attributes = categorical_attributes
        self._ordinal_attributes = ordinal_attributes
        self._scale = scale
        self._unscale = unscale

        # Rebuild the model with an input modification layer
        input_mod_layer = InputAddLayer()
        self.input = self.original_model.layers[0].input
        model = input_mod_layer(self.original_model.layers[0].output)
        for layer in self.original_model.layers[1:]:
            model = layer(model)
        self.output = model

        self.model = Model(inputs=self.input, outputs=self.output)
        # Only allow the input modification layer to be trained
        for layer in self.model.layers[2:]:
            layer.trainable = False

        self.input_modifier = input_mod_layer

        desired = Input(shape=[self.output.shape[1]])
        desired_loss = -K.sum(desired * K.log(self.output), axis=1)
        grad = K.gradients(desired_loss, [self.input])
        self.adv_grad = K.function([self.input, desired], [grad[0]])

    def train(self, input, target, num_classes, num_changed_attributes=None, max_epochs=1000, skip_attributes=0,
        categorical_threshold=0, direction_constraints=None):
        '''
        Train the model to produce a CADEX explanation.
        :param input: original unmodified input vector, as a Pandas dataframe
        :param target: target class index
        :param num_changed_attributes: maximum number of attributes to allow to change
        :param max_epochs: limit number of epochs in case a solution is not found
        :param adjust_categorical: if true, ensure categorical variables don't violate constraints, and flip when
           they go over the threshold
        :param category_map: map from original category column name to list of one-hot encoded columns
        :param skip_attributes: number of attributes to skip, sorted by magnitude of the gradient
        :param categorical_threshold: threshold above which to flip categorical attributes
        :param direction_mask: mask vector to limit search direction (see paper for more details)
        :param ordinal_attributes: list of ordinal attribute column names, used to ensure attribute values are integers
        :return: tuple of [result, epoch] where result is the adversarial example found at epoch
        '''
        self.reset()
        self._begin_train(input, target, num_changed_attributes=num_changed_attributes,
            skip_attributes=skip_attributes, direction_mask=direction_constraints)

        return self._train_step(input, target, num_classes, max_epochs=max_epochs, categorical_threshold=categorical_threshold)

    def _begin_train(self, input, target, num_changed_attributes=None, skip_attributes=0,
        direction_mask=None):
        '''
        Initializes the training process and prepares the model.
        Calculates the initial pre-training gradient, and determines the mask to apply
        '''
        mask = np.ones(input.shape)
        if num_changed_attributes is not None or direction_mask is not None:
            first_grad = self.get_gradient(input, target)

            # There's no point to decrease the values of categorical attributes, or to increase the value of a
            # categorical attribute when the original value is 1. We can just zero the gradient in such cases.
            if self._categorical_attributes is not None:
                categorical_attributes = [j for i in self._categorical_attributes for j in i]
                cols = list(input.columns)
                for attr_set in self._categorical_attributes:
                    for attr in attr_set:
                        if first_grad[0, cols.index(attr)] > 0 or input[attr].values[0] > 0:
                            first_grad[0, cols.index(attr)] = 0

            # If we need to skip attributes, zero the gradient in the top K indices
            if skip_attributes > 0:
                ind = np.argsort(np.abs(first_grad))
                first_grad[0, ind[0, -skip_attributes:]] = 0

            # If direction mask is specified, apply it to the mask
            if direction_mask is not None:
                mask = mask * np.sign(np.sign(first_grad * (-direction_mask)) + 1)

            # If we limit the number of attributes, make a mask which is 1 in the indices of the top K attributes
            # and 0 elsewhere
            if num_changed_attributes is not None:
                ind = np.argsort(np.abs(first_grad * mask))
                change_mask = np.zeros(input.shape)
                change_mask[0, ind[0, -num_changed_attributes:]] = 1
                mask = mask * change_mask

        # Initialize the Adam optimizer, giving it the calculated input mask
        adam = MaskedAdam(K.constant(mask, dtype='float32'))

        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    def _train_step(self, input, target, num_classes, max_epochs=1000, categorical_threshold=0):
        '''
        Performs the training to find the adversarial example.
        For each epoch, trains the model on a single observation, which is the input
        Then, adjusts categorical attributes if needed.
        Finally, a check is made to see whether the input now classifies as the target class and the algorithm can
        terminate.
        '''
        res = None
        categorical_attributes_index = None
        ordinal_attributes_index = None
        pred_threshold = 1 / num_classes
        cols = list(input.columns)
        if self._categorical_attributes is not None:
            categorical_attributes_index = [[cols.index(j) for j in i] for i in self._categorical_attributes]
        if self._ordinal_attributes is not None:
            ordinal_attributes_index = [cols.index(i) for i in self._ordinal_attributes]

        for epoch in range(max_epochs):
            self.model.train_on_batch(input, np.array([np.eye(self.output.shape[1])[target]]))

            if self._categorical_attributes is not None:
                self._adjust_categorical(input, categorical_attributes_index, threshold=categorical_threshold)

            # Check if input classifies as target
            if self.model.predict(input)[0, target] > pred_threshold:
                if self._categorical_attributes is not None or self._ordinal_attributes is not None:
                    # The categorical attributes may be in a non-clipped state, i.e. not exactly 1 and 0.
                    # Perform clipping, then test the condition again. If the input still gets the target classification,
                    # save the clipped attribute values and return. Otherwise restore the original weights.
                    constrained_input, constrained_weights = self._apply_constraints(input, categorical_attributes_index,
                        ordinal_attributes_index)
                    if self.original_model.predict(constrained_input)[0, target] > pred_threshold:
                        self.input_modifier.set_weights([constrained_weights])
                        if isinstance(input, pd.DataFrame):
                            res = pd.DataFrame(constrained_input, columns=input.columns, index=input.index)
                        else:
                            res = constrained_input
                        break
                else:
                    res = self.input_modifier.transform(input)
                    break

        return res, epoch

    def _adjust_categorical(self, input, categorical_attributes, threshold=0):
        input_mod = self.transform(input)
        input_target = input_mod.copy()

        # for each set of one-hot encoded attributes
        for attr_set in categorical_attributes:
            max_vals = np.argsort(input_mod[0, attr_set])[::-1]
            # check if the second highest attribute is above the threshold, and if so set it to one
            if len(max_vals)>1:
              if input_mod[0, attr_set[max_vals[1]]] > threshold:
                  input_target[0, attr_set] = 0
                  input_target[0, attr_set[max_vals[1]]] = 1

        if np.any(input_target != input_mod):
            updated_weights = input_target - input
            self.input_modifier.set_weights([updated_weights])
        return input_target

    def _apply_constraints(self, input, categorical_attributes, ordinal_attributes):
        input_mod = self.transform(input)
        input_target = input_mod.copy()

        if categorical_attributes is not None:
            for attr_set in categorical_attributes:
                max_vals = np.argsort(input_mod[0, attr_set])[::-1]
                max_cat = attr_set[max_vals[0]]
                # zero all but the highest attribute
                input_target[0, attr_set] = 0
                input_target[0, max_cat] = 1

        if ordinal_attributes is not None:
            unscaled = self._unscale(input_target)
            for attr in ordinal_attributes:
                unscaled[0, attr] = np.round(unscaled[0, attr])
            input_target = self._scale(unscaled)

        updated_weights = input_target - input
        return input_target, updated_weights

    def get_gradient(self, input, target):
        '''
        Calculate the gradient in input space
        :param input: input vector
        :param target: target class index
        :return: gradient vector
        '''
        return self.adv_grad([input, [np.eye(self.output.shape[1])[target]]])[0]

    def transform(self, input):
        return self.input_modifier.transform(input)

    def reset(self):
        K.set_value(self.input_modifier.weights[0], np.zeros(self.input_modifier.get_weights()[0].shape))
        # K.set_value(self.input_modifier.weights[0], np.zeros(K.get_variable_shape(self.input_modifier.weights[0])))

class InputAddLayer(Layer):
    '''
    Input modification layer.
    Has a vector of trainable weights which are added to the incoming input.
    Subclasses Keras' Layer class and implements 3 abstract methods
    '''
    def build(self, input_shape):
        self._weights = self.add_weight(name='weights', shape=(1, input_shape[1]), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        return x + self._weights

    def compute_output_shape(self, input_shape):
        return input_shape

    def transform(self, input):
        res = input + self.get_weights()[0]
        if isinstance(res, pd.DataFrame):
            return res.values

class MaskedAdam(Adam):
    '''
    Adam optimizer, with gradient mask.
    The mask is given in the constructor, and used to multiply by the gradients.
    get_gradients from the base Optimizer class is overridden to implement this behaviour.
    '''
    def __init__(self, mask, **kwargs):
        self._mask = mask
        super().__init__(**kwargs)

    def get_gradients(self, loss, params):
        return [i * self._mask for i in super().get_gradients(loss, params)]