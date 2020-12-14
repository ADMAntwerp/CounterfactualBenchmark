import jsonlines
import matplotlib.pyplot as plt
# import matplotlib.patches as pt
import random
import numpy as np
import tensorflow as tf
import functools
import os
# from magenta.models.sketch_rnn.utils import *
# import svgwrite
# import time
import copy

cur_dir = os.path.dirname(__file__)
data_path = os.path.join(cur_dir, 'full_simplified_cat.ndjson')
gold_idx = 9384
max_line_num = 128
IMG_WIDTH = 256
non_line = 256


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def prune(strokes):
    ret = []
    for s in strokes:
        if any(p > 255 for p in s):
            continue
        ret.append(s)
    return ret


def find_gold():
    with jsonlines.open(data_path) as reader:
        for (idx, obj) in enumerate(reader):
            if idx == gold_idx:
                return obj['drawing']


def separate_data():
    drawings = []
    gold_drawing = None
    with jsonlines.open(data_path) as reader:
        for (idx, obj) in enumerate(reader):
            if idx == gold_idx:
                gold_drawing = obj['drawing']
            else:
                strokes = obj['drawing']
                if count_line_num_in_stroke(strokes) <= max_line_num:
                    drawings.append(obj['drawing'])

    # split drawings into train, validate, and test
    length = len(drawings)
    print('Total number of negative trainings: ' + str(length))
    train_end_idx = int(length * 0.5)
    val_end_idx = int(length * 0.75)
    print('Train: 0-' + str(train_end_idx - 1))
    print('Validate: ' + str(train_end_idx) + '-' + str(val_end_idx - 1))
    print('Test: ' + str(val_end_idx) + '-' + str(length - 1))
    train = drawings[0:train_end_idx]
    val = drawings[train_end_idx:val_end_idx]
    test = drawings[val_end_idx:length]
    return gold_drawing, train, val, test


def create_instance(drawing):
    lines = convert_to_ordered_lines(drawing)
    while (len(lines) < max_line_num):
        lines.append([non_line, non_line, non_line, non_line])
    assert (len(lines) == max_line_num)
    return lines


def del_nth_stroke(strokes, n):
    ret = copy.deepcopy(strokes)
    ret.pop(n)
    return ret


def del_random_stroke(strokes, max_del_len=1000):
    candiate = []
    for i in range(len(strokes)):
        if len(strokes[i][0]) <= max_del_len:
            candiate.append(i)

    del_can = random.sample(candiate, 1)[0]

    ret = []
    for idx, v in enumerate(strokes):
        if idx != del_can:
            ret.append(v)

    return ret


def create_varied_del_instance(drawing):
    drawing = randomize(drawing)
    lines = convert_to_lines(drawing)
    sample_size = random.randint(85, len(lines) - 1)
    lines = random.sample(lines, sample_size)
    cur_length = len(lines)
    # add_num = random.randint(0, max_line_num - cur_length)
    add_num = 0
    for i in range(add_num):
        x1 = random.randint(1, 255)
        y1 = random.randint(1, 255)
        x2 = random.randint(1, 255)
        y2 = random.randint(1, 255)
        lines.append([x1, y1, x2, y2])
    lines.sort(key=functools.cmp_to_key(line_cmp))
    while (len(lines) < max_line_num):
        lines.append([non_line, non_line, non_line, non_line])
    assert (len(lines) == max_line_num)
    return lines


def scale(strokes, min_scale=0.5, clip_min=1, clip_max=255):
    ret = []
    scale = random.uniform(min_scale, 0.5)
    for s in strokes:
        s1 = [[], []]
        for x, y in zip(s[0], s[1]):
            x *= scale
            y *= scale

            x = np.clip(x, clip_min, clip_max)
            y = np.clip(y, clip_min, clip_max)

            s1[0].append(x)
            s1[1].append(y)

        ret.append(s1)
    return ret


def drop(strokes, drop_freq=0.1):
    ret = []
    for s in strokes:
        s1 = [[], []]
        for x, y in zip(s[0], s[1]):
            if random.random() < drop_freq:
                continue

            s1[0].append(x)
            s1[1].append(y)

        ret.append(s1)
    return ret


def create_varied_instance(drawing):
    drawing = randomize(drawing)
    return create_instance(drawing)


def create_bad_varied_instance(drawing):
    drawing = randomize(drawing, step_max=5, wrong_step_min=10, wrong_step_max=20,
                        wrong_max_perm=10)
    return create_instance(drawing)


def create_batch1(batch_size, gold, neg_samples, mixed_missing=False):
    xs = []
    ys = []
    for i in range(int(batch_size / 2)):
        xs.append(normalize(create_varied_instance(gold)))
        ys.append([0, 1])

    if mixed_missing:
        neg = int(batch_size / 6)
        miss = int(batch_size / 6)
        bad = int(batch_size / 6)
        # neg = 0
        # miss = 0
        # bad = int(batch_size/2)
    else:
        neg = int(batch_size / 2)
        miss = 0

    for i in range(neg):
        xs.append(
            normalize(create_varied_instance(neg_samples[random.randint(0, len(neg_samples) - 1)])))
        ys.append([1, 0])

    for i in range(miss):
        del_gold = create_varied_del_instance(gold)
        xs.append(normalize(del_gold))
        ys.append([1, 0])

    gold_inst = create_instance(gold)
    for i in range(bad):
        bad_var = create_bad_varied_instance(gold)
        xs.append(normalize(bad_var))
        ys.append([1, 0])

    p = np.random.permutation(len(xs))

    return np.array(xs)[p], np.array(ys)[p]


def create_varied_instance2(drawing):
    drawing = randomize(drawing)
    drawing = scale(drawing)
    drawing = drop(drawing)
    return create_instance(drawing)


def create_batch2(batch_size, gold, neg_samples, mixed_missing=False):
    xs = []
    ys = []
    for i in range(int(batch_size / 2)):
        xs.append(normalize(create_varied_instance(gold)))
        ys.append([0, 1])

    if mixed_missing:
        # neg = int(batch_size / 4)
        miss = int(batch_size / 2)
        neg=0
    else:
        neg = int(batch_size / 2)
        miss = 0

    for i in range(neg):
        xs.append(
            normalize(create_varied_instance(neg_samples[random.randint(0, len(neg_samples) - 1)])))
        ys.append([1, 0])

    for i in range(miss):
        del_gold = create_varied_del_instance(gold)
        xs.append(normalize(del_gold))
        ys.append([1, 0])

    p = np.random.permutation(len(xs))

    return np.array(xs)[p], np.array(ys)[p]


def create_batch(batch_size, gold, neg_samples, mixed_missing=False):
    xs = []
    ys = []
    for i in range(int(batch_size / 2)):
        xs.append(normalize(create_varied_instance(gold)))
        ys.append([0, 1])

    if mixed_missing:
        neg = int(batch_size / 4)
        miss = int(batch_size / 4)
    else:
        neg = int(batch_size / 2)
        miss = 0

    for i in range(neg):
        xs.append(
            normalize(create_varied_instance(neg_samples[random.randint(0, len(neg_samples) - 1)])))
        ys.append([1, 0])

    for i in range(miss):
        del_gold = del_random_stroke(gold)
        xs.append(normalize(create_varied_instance(del_gold)))
        ys.append([1, 0])

    p = np.random.permutation(len(xs))

    return np.array(xs)[p], np.array(ys)[p]


def draw_matrix(matrix, fig_name):
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            x1 = x
            y1 = y
            x2 = matrix[x1][y1][0]
            y2 = matrix[x1][y1][1]
            if x2 == -1:
                continue
            plt.plot([x1, x2], [y1, y2])

    plt.xlim(xmin=0, xmax=256)
    plt.ylim(ymin=0, ymax=256)

    plt.gca().invert_yaxis()

    plt.savefig(fig_name + '.pdf', bbox_inches='tight')
    # plt.show()
    plt.clf()


def draw_lines(lines, fig_name):
    for l in lines:
        plt.plot([l[0], l[2]], [l[1], l[3]], 'b-')

    plt.xlim(xmin=0, xmax=256)
    plt.ylim(ymin=0, ymax=256)

    plt.gca().invert_yaxis()

    plt.savefig(fig_name + '.png', bbox_inches='tight')
    # plt.show()
    plt.clf()


def draw_lines2(lines, lines2, fig_name):
    for l in lines:
        plt.plot([l[0], l[2]], [l[1], l[3]], 'b-')
    for l in lines2:
        plt.plot([l[0], l[2]], [l[1], l[3]], 'r-')

    plt.xlim(xmin=0, xmax=256)
    plt.ylim(ymin=0, ymax=256)

    plt.gca().invert_yaxis()

    plt.savefig(fig_name + '.png', bbox_inches='tight')
    # plt.show()
    plt.clf()


def draw_lines_and_boxes(lines, boxes, change_indices, fig_name):
    for i in range(len(lines)):
        if i in change_indices:
            continue
        l = lines[i]
        plt.plot([l[0], l[2]], [l[1], l[3]], 'r-')

    sub_lines = [lines[i] for i in change_indices]

    for l in sub_lines:
        plt.plot([l[0], l[2]], [l[1], l[3]], 'c-')

    for b in boxes:
        xlb, xub, ylb, yub = b
        # xlb,ylb -> xub, ylb
        plt.plot([xlb, xub], [ylb, ylb], 'b-')
        # xub, ylb -> xub, yub
        plt.plot([xub, xub], [ylb, yub], 'b-')
        # xlb,ylb -> xlb, yub
        plt.plot([xlb, xlb], [ylb, yub], 'b-')
        # xlb,yub -> xub, yub
        plt.plot([xlb, xub], [yub, yub], 'b-')

    plt.xlim(xmin=0, xmax=256)
    plt.ylim(ymin=0, ymax=256)

    plt.gca().invert_yaxis()

    plt.savefig(fig_name + '.png', bbox_inches='tight')
    # plt.show()
    plt.clf()


def normalize(lines):
    return np.array(lines) / 256.0


def randomize(strokes, step_max=5, clip_min=1, clip_max=255, wrong_step_min=None,
              wrong_step_max=None, wrong_max_perm=1000):
    ret = []
    wrong_perm_indices = None
    if wrong_step_min is not None and wrong_step_max is not None:
        if len(strokes) > wrong_max_perm:
            num_points = 0
            for s in strokes:
                num_points += len(s[0])
            indices = list(range(num_points))
            wrong_perm_indices = random.sample(indices, random.randint(1, wrong_max_perm))

    pidx = 0
    for s in strokes:
        s1 = [[], []]
        for x, y in zip(s[0], s[1]):
            if wrong_perm_indices is not None and pidx in wrong_perm_indices:
                if bool(random.getrandbits(1)):
                    x = x + random.randint(wrong_step_min, wrong_step_max)
                else:
                    x = x - random.randint(wrong_step_min, wrong_step_max)

                if bool(random.getrandbits(1)):
                    y = y + random.randint(wrong_step_min, wrong_step_max)
                else:
                    y = y - random.randint(wrong_step_min, wrong_step_max)
            else:
                x = x + random.randint(-step_max, step_max)
                y = y + random.randint(-step_max, step_max)

            x = np.clip(x, clip_min, clip_max)
            y = np.clip(y, clip_min, clip_max)

            s1[0].append(x)
            s1[1].append(y)
            pidx += 1

        ret.append(s1)
    return ret


def cal_loc(x, y, h_img_w):
    if not ((0 <= x < 2 * h_img_w) and (0 <= y < 2 * h_img_w)):
        print('Haha')
    assert (0 <= x < 2 * h_img_w)
    assert (0 <= y < 2 * h_img_w)
    if x < h_img_w and y < h_img_w:
        return 1
    if x < h_img_w and y >= h_img_w:
        return 2
    if x >= h_img_w and y >= h_img_w:
        return 3
    if x >= h_img_w and y < h_img_w:
        return 4


def point_cmp(x1, y1, x2, y2):
    if x1 == x2 and y2 == y1:
        return 0
    half_IMG_W = IMG_WIDTH / 2
    while True:
        loc1 = cal_loc(x1, y1, half_IMG_W)
        loc2 = cal_loc(x2, y2, half_IMG_W)
        if loc1 != loc2:
            return loc1 - loc2
        x1 = x1 % half_IMG_W
        y1 = y1 % half_IMG_W
        x2 = x2 % half_IMG_W
        y2 = y2 % half_IMG_W
        half_IMG_W = half_IMG_W / 2


def line_cmp(l1, l2):
    cmp1 = point_cmp(l1[0], l1[1], l2[0], l2[1])
    if cmp1 != 0:
        return cmp1
    return point_cmp(l1[2], l1[3], l2[2], l2[3])


def count_line_num_in_stroke(strokes):
    ret = 0
    for s in strokes:
        ret += (len(s[0]) - 1)
    return ret


def convert_to_lines(strokes):
    ret = []
    for s in strokes:
        for idx in range(len(s[0]) - 1):
            x1 = int(round(s[0][idx]))
            y1 = int(round(s[1][idx]))
            x2 = int(round(s[0][idx + 1]))
            y2 = int(round(s[1][idx + 1]))
            if x1 > x2 or (x1 == x2 and y1 > y2):
                t = x1
                x1 = x2
                x2 = t
                t = y1
                y1 = y2
                y2 = t
            ret.append([x1, y1, x2, y2])

    return ret


def convert_to_ordered_lines(strokes):
    ret = convert_to_lines(strokes)
    ret.sort(key=functools.cmp_to_key(line_cmp))
    return ret


def convert_to_matrix(strokes, size=256):
    ret = np.full([size, size, 2], -1)
    for s in strokes:
        for idx in range(len(s[0]) - 1):
            x1 = s[0][idx]
            y1 = s[1][idx]
            if ret[x1][y1][0] != -1:
                deltas = []
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        deltas.append((i, j))

                for d in deltas:
                    x1 += d[0]
                    y1 += d[1]
                    if ret[x1][y1][0] == -1:
                        break

            if ret[x1][y1][0] != -1:
                continue

            x2 = s[0][idx + 1]
            y2 = s[1][idx + 1]
            ret[x1][y1][0] = x2
            ret[x1][y1][1] = y2

    return ret


def convert_abs_to_rel(drawing):
    """Convert polyline format to stroke-3 format."""
    eos = 0
    strokes = []
    for line in drawing:
        linelen = len(line[0])
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[0][i], line[1][i], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :]


def convert_rel_to_abs(strokes, start_x=0, start_y=0):
    x = start_x
    y = start_y
    lines = []
    line = [[x], [y]]
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line[0].append(x)
            line[1].append(y)
            lines.append(line)
            line = [[], []]
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line[0].append(x)
            line[1].append(y)

    return lines


# # little function that displays vector images and saves them to .svg
# def draw_strokes(data, factor=0.2, svg_filename='/tmp/sketch_rnn/svg/sample.svg'):
#     tf.gfile.MakeDirs(os.path.dirname(svg_filename))
#     min_x, max_x, min_y, max_y = get_bounds(data, factor)
#     dims = (50 + max_x - min_x, 50 + max_y - min_y)
#     dwg = svgwrite.Drawing(svg_filename, size=dims)
#     dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))
#     lift_pen = 1
#     abs_x = 25 - min_x
#     abs_y = 25 - min_y
#     p = "M%s,%s " % (abs_x, abs_y)
#     command = "m"
#     for i in range(len(data)):
#         if (lift_pen == 1):
#             command = "m"
#         elif (command != "l"):
#             command = "l"
#         else:
#             command = ""
#         x = float(data[i, 0]) / factor
#         y = float(data[i, 1]) / factor
#         lift_pen = data[i, 2]
#         p += command + str(x) + "," + str(y) + " "
#     the_color = "black"
#     stroke_width = 1
#     dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))
#     dwg.save()


# generate a 2D grid of many vector drawings
def make_grid_svg(s_list, grid_space=10.0, grid_space_x=16.0):
    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start = x[0]
        x_end = x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max = x.max(axis=0)
        x_min = x.min(axis=0)
        center_loc = (x_max + x_min) * 0.5
        return x_start - center_loc, x_end

    x_pos = 0.0
    y_pos = 0.0
    result = [[x_pos, y_pos, 1]]
    for sample in s_list:
        s = sample[0]
        grid_loc = sample[1]
        grid_y = grid_loc[0] * grid_space + grid_space * 0.5
        grid_x = grid_loc[1] * grid_space_x + grid_space_x * 0.5
        start_loc, delta_pos = get_start_and_end(s)

        loc_x = start_loc[0]
        loc_y = start_loc[1]
        new_x_pos = grid_x + loc_x
        new_y_pos = grid_y + loc_y
        result.append([new_x_pos - x_pos, new_y_pos - y_pos, 0])

        result += s.tolist()
        result[-1][2] = 1
        x_pos = new_x_pos + delta_pos[0]
        y_pos = new_y_pos + delta_pos[1]
    return np.array(result)


def main():
    strokes = None
    max_lines = -1
    drawing = None
    length_map = {}
    gold_line_num = None
    total_num = 0
    under150 = 0
    with jsonlines.open(cur_dir + '/full_simplified_cat.ndjson') as reader:
        for (idx, obj) in enumerate(reader):
            strokes = obj['drawing']
            lines = convert_to_ordered_lines(strokes)
            len1 = len(lines)
            if len1 in length_map:
                length_map[len1] += 1
            else:
                length_map[len1] = 1
            if len1 > max_lines:
                max_lines = len1
                drawing = lines
            if idx == 9384:
                gold_line_num = len1
            else:
                total_num += 1
                if len1 < 120:
                    under150 += 1

    items = list(length_map.items())

    items.sort()

    print(items)

    print('Gold number: ' + str(gold_line_num))

    print('Number of lines: ' + str(max_lines))
    draw_lines(drawing, 'many_lines')

    print(str(under150) + ' out of ' + str(total_num))

    # strokes = randomize(strokes)
    #
    # draw_strokes(strokes, 'ori')
    #
    # lines = convert_to_ordered_lines(strokes)
    #
    # draw_lines(lines, 'lines')


if __name__ == '__main__':
    import json, sys
    idx = int(sys.argv[1])
    data = np.load(os.path.join(os.path.dirname(__file__),'cat.missing.npz'))
    instances = data['data']
    instance = instances[idx] * 255
    solution = json.load(open('run_quickdraw_quickdraw_%d.json' % idx, 'r'))
    sol = solution['output']['best_result']
    if sol['cost'] is not None:
        end = np.array(sol['final_instance']) * 255
        for k, line in enumerate(end):
            if np.sum(instance[k] - line) == 0:
                end[k, :] = [0., 0., 0., 0.]
        draw_lines2(instance, end,
                    'solution_%d_best_L=%d' % (idx, len(sol['p'])))
    for j, sol in enumerate(solution['output']['history']):
        if sol['cost'] is not None:
            end = np.array(sol['final_instance']) * 255
            for k, line in enumerate(end):
                if np.sum(instance[k] - line) == 0:
                    end[k, :] = [0., 0., 0., 0.]

            draw_lines2(instance, end, 'solution_%d_L=%d' % (idx, len(sol['p'])))

