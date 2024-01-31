import matplotlib.pyplot as plt
import random
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def load_dataset():
    meta = unpickle(f'cifar-10-batches-py//batches.meta')
    labels = []
    data = []

    for i in range(1,6):
        dataset = unpickle(f'cifar-10-batches-py//data_batch_{i}')
        labels.append(dataset[b'labels'])
        data.append(dataset[b'data'])

    test_dataset = unpickle(f'cifar-10-batches-py//test_batch')

    test_data = test_dataset[b'data'].reshape(-1,3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = np.array(test_dataset[b'labels'])


    # Konvertieren Sie die Listen von numpy Arrays in ein einziges numpy Array
    train_data = np.concatenate(data, axis=0).reshape(-1,3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.concatenate(labels, axis=0)
    return train_data, train_labels, test_data, test_labels, meta, 


if __name__ == '__main__':
    try:
        train_data, train_labels, test_data, test_labels, meta = load_dataset()
        while True:
            randpos = random.randint(0, len(train_labels))
            example_label = train_labels[randpos]
            example_data = train_data[randpos]
            pos_1_example_data = example_data

            # Für pos_1_example_data
            label_names = meta[b'label_names']
            print(label_names[example_label])
            plt.imshow(pos_1_example_data)
            plt.show()
    except KeyboardInterrupt:
        print('finished')
        pass

