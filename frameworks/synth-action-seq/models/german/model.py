import tensorflow as tf
from common.paths import GERMAN_DIR
from models.german.dataset import GermanDataset

# Layers=2, Units=40, Dropout=0.5
# loss: 0.4229711484909058
# acc: 0.79

LEARNING_RATE = 1e-4
LAYERS = 2
UNITS = 40
DROPOUT = 0.7
BATCH_SIZE = 100
NUM_EPOCHS = 1000
TRUE_LABEL = [1., 0.]
FALSE_LABEL = [0., 1.]


class GermanCredit:
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
        input_dim = train_data.shape[1]
        label_dim = train_labels.shape[1]
        self.input_dim = input_dim
        self.model = tf.keras.models.Sequential()
        for i in range(LAYERS):
            input_dim = UNITS if i > 0 else input_dim
            self.model.add(tf.keras.layers.Dense(UNITS, input_dim=input_dim, activation='relu'))
            self.model.add(tf.keras.layers.Dropout(DROPOUT))
        self.model.add(tf.keras.layers.Dense(label_dim))

        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

        self.model.compile(
            loss=self.loss_fn,
            optimizer=optimizer, metrics=['accuracy'])
        self.model.fit(train_data, train_labels, epochs=epochs, shuffle=True,
                       batch_size=batch_size)

    def restore(self, weights_file):
        self.model = tf.keras.models.load_model(weights_file,
                                                custom_objects=dict(loss_fn=self.loss_fn))
        self.input_dim = self.model.get_layer(index=0).input_shape[1]


    def save(self, file_name):
        tf.keras.models.save_model(self.model, file_name)

    def predict(self, input):
        return self.model.predict(input)

    def evaluate(self, data, labels):
        metrics = self.model.evaluate(data, labels)
        for name, val in zip(self.model.metrics_names, metrics):
            print('%s: %s' % (name, val))


if __name__ == '__main__':
    train_file = GERMAN_DIR / 'german_z.train.npy'
    test_file = GERMAN_DIR / 'german_z.test.npy'
    model_ckpt = GERMAN_DIR / 'model.h5'
    if not train_file.exists():
        raise ValueError(
            'german_z.train.npy does not exist. Please run python -m models.german.dataset')
    if not test_file.exists():
        raise ValueError(
            'german_z.test.npy does not exist. Please run python -m models.german.dataset')
    model = GermanCredit()
    train_dataset = GermanDataset(str(train_file))
    test_dataset = GermanDataset(str(test_file))
    model.train(train_dataset.data, train_dataset.labels)
    model.evaluate(test_dataset.data, test_dataset.labels)
    model.save(str(model_ckpt))
