from models.quickdraw.common import *

batch_size = 32
max_steps = 10000
test_size = 1000


def main():
    gold, train, val, test = separate_data()
    # batch = [np.zeros((32 * max_steps, 128, 4)), np.zeros((32 * max_steps, 2))]
    # for i in range(max_steps):
    #     print(str(i) + "/" + str(max_steps))
    #     new_batch = create_batch2(batch_size, gold, train, True)
    #     batch[0][32*i:32*i+32, :, :] = new_batch[0]
    #     batch[1][32*i:32*i+32, :] = new_batch[1]
    # np.savez('./cat.train.npz', data=batch[0], labels=batch[1])
    # print(batch[0].shape)
    test_batch = create_batch2(test_size, gold, test, True)
    np.savez('./local_pack/models/quickdraw/cat.missing.npz', data=test_batch[0], labels=test_batch[1])
    print(test_batch[0].shape)


if __name__ == '__main__':
    main()
