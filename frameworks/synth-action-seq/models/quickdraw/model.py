import tensorflow as tf
from common.paths import QUICKDRAW_DIR
from models.quickdraw.dataset import QuickdrawDataset

# loss: 0.017243763625621796
# acc: 0.9935

LEARNING_RATE = 1e-2
BATCH_SIZE = 32
NUM_EPOCHS = 1
TRUE_LABEL = [0., 1.]
FALSE_LABEL = [1., 0.]


class QuickDraw:
    def __init__(self):
        self.model = None
        self.TRUE_LABEL = TRUE_LABEL
        self.FALSE_LABEL = FALSE_LABEL

    def __call__(self, instance_tensor):
        if self.model is None:
            raise ValueError('Model has not been configured')
        return self.model(instance_tensor)

    def loss_fn(self, correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=correct,
                                                          logits=predicted)

    def train(self, train_data, train_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
        label_dim = train_labels.shape[1]
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='same',
                                              kernel_initializer=tf.keras.initializers.random_normal(
                                                  stddev=0.1),
                                              bias_initializer=tf.keras.initializers.Constant(0.1),
                                              activation=tf.keras.activations.relu, input_shape=(128, 4)))
        self.model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='same',
                                              kernel_initializer=tf.keras.initializers.random_normal(
                                                  stddev=0.1),
                                              bias_initializer=tf.keras.initializers.Constant(0.1),
                                              activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=1, padding='same',
                                              kernel_initializer=tf.keras.initializers.random_normal(
                                                  stddev=0.1),
                                              bias_initializer=tf.keras.initializers.Constant(0.1),
                                              activation=tf.keras.activations.relu))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu,
                                             kernel_initializer=tf.keras.initializers.random_normal(
                                                 stddev=0.1),
                                             bias_initializer=tf.keras.initializers.Constant(0.1)))
        self.model.add(tf.keras.layers.Dropout(rate=0.1))
        self.model.add(tf.keras.layers.Dense(label_dim, activation=tf.keras.activations.relu,
                                             kernel_initializer=tf.keras.initializers.random_normal(
                                                 stddev=0.1),
                                             bias_initializer=tf.keras.initializers.Constant(0.1)))

        optimizer = tf.keras.optimizers.Adadelta(LEARNING_RATE)
        print(self.model.summary())
        #
        # self.model.compile(
        #     loss=self.loss_fn,
        #     optimizer=optimizer, metrics=['accuracy'])
        # print(self.model.summary())
        # self.model.fit(train_data, train_labels, epochs=epochs, shuffle=True,
        #                batch_size=batch_size)

    def restore(self, weights_file):
        self.model = tf.keras.models.load_model(weights_file,
                                                custom_objects=dict(loss_fn=self.loss_fn))

    def save(self, file_name):
        tf.keras.models.save_model(self.model, file_name)

    def predict(self, input):
        return self.model.predict(input)

    def evaluate(self, data, labels):
        metrics = self.model.evaluate(data, labels)
        for name, val in zip(self.model.metrics_names, metrics):
            print('%s: %s' % (name, val))


if __name__ == '__main__':
    train_file = QUICKDRAW_DIR / 'cat.train.npz'
    test_file = QUICKDRAW_DIR / 'cat.test.npz'
    model_ckpt = QUICKDRAW_DIR / 'model.h5'
    if not train_file.exists():
        raise ValueError(
            'quickdraw_z.train.npy does not exist. Please run python -m models.quickdraw.dataset')
    if not test_file.exists():
        raise ValueError(
            'quickdraw_z.test.npy does not exist. Please run python -m models.quickdraw.dataset')
    model = QuickDraw()
    train_dataset = QuickdrawDataset(str(train_file))
    test_dataset = QuickdrawDataset(str(test_file))
    model.train(train_dataset.data, train_dataset.labels)
    # model.evaluate(test_dataset.data, test_dataset.labels)
    # model.save(str(model_ckpt))
