from models.quickdraw.common import *
import os

IMAGE_WIDTH = 256
train_num = 70000
val_num = 2500
test_num = 2500


def main():
    gold_drawing, train, val, test = separate_data()
    train_data = []
    validation_data = []
    test_data = []

    for i in range(train_num):
        drawing = randomize(gold_drawing)
        random.shuffle(drawing)
        train_data.append(convert_abs_to_rel(drawing))

    for i in range(val_num):
        drawing = randomize(gold_drawing)
        random.shuffle(drawing)
        validation_data.append(convert_abs_to_rel(drawing))

    for i in range(test_num):
        drawing = randomize(gold_drawing)
        random.shuffle(drawing)
        test_data.append(convert_abs_to_rel(drawing))

    filename = os.path.join('./', 'gold_cat.npz')
    np.savez_compressed(filename, train=train_data, valid=validation_data, test=test_data)


if __name__ == '__main__':
    main()