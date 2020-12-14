import numpy as np
import struct
from struct import unpack


def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))
    return {
        'recognized': recognized,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


def stroke2segments(stroke):
    xs, ys = stroke
    prev = None
    segments = []
    for x, y in zip(xs, ys):
        if prev is None:
            prev = (x, y)
        else:
            x1, y1 = prev
            x2, y2 = x, y
            segments += [x1, y1, x2, y2]
    return segments


def strokes2segments(strokes):
    segments = []
    for stroke in strokes:
        segments += stroke2segments(stroke)
    return segments


sample = None
max_segments = 0
count = 0
images = []
for drawing in unpack_drawings('full_binary_cat.bin'):
    segments = strokes2segments(drawing['image'])
    if len(segments) > 512:
        count += 1
    else:
        image = dict(data=segments, label=[0, 1] if drawing['recognized'] else [1, 0])
        images.append(image)

num_imgs = len(images)
processed_data = np.ndarray(shape=(num_imgs, 512), dtype=int)
processed_labels = np.ndarray(shape=(num_imgs, 2), dtype=int)

for i in range(num_imgs):
    image = images[i]
    segments = image['data']
    processed_data[i][:len(segments)] = np.array(segments)
    processed_labels[i][:] = np.array(image['label'])


print(processed_labels.shape)
print(processed_data.shape)
print(processed_data[0])

np.savez('processed_cats.npz', data=processed_data, labels=processed_labels)
