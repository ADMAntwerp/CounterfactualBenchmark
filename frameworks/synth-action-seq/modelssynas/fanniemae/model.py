import tensorflow as tf

FALSE_LABEL = [0., 1.]
TRUE_LABEL = [1., 0.]


class FannieMae:
    def __init__(self):
        self.model = None
        self.TRUE_LABEL = TRUE_LABEL
        self.FALSE_LABEL = FALSE_LABEL

    def __call__(self, instance_tensor):
        if self.model is None:
            raise ValueError('Model has not been configured')
        return self.model(instance_tensor)

    def loss_fn(self, correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=correct, logits=predicted)

    def restore(self, weights_file):
        self.model = tf.keras.models.load_model(weights_file,custom_objects=dict(loss_fn=self.loss_fn))
        self.input_dim = self.model.get_layer(index=0).input_shape[1]

    def predict(self, input):
        return self.model.predict(input)

    def evaluate(self, data, labels):
        metrics = self.model.evaluate(data, labels)
        for name, val in zip(self.model.metrics_names, metrics):
            print('%s: %s' % (name, val))