import argparse
import tensorflow as tf
import os
import sys
import numpy as np
import yaml
from tqdm import tqdm

from utils import gen_default_boxes
from utils import decode, calc_nms
from data_utils import create_batch_generator
from image_utils import ImageVisualizer
from SSD import create_ssd
from PIL import Image

def hard_negative_mining(loss, gt_confs, neg_ratio):
    """ Hard negative mining algorithm
        to pick up negative examples for back-propagation
        base on classification loss values
    Args:
        loss: list of classification losses of all default boxes (B, num_default)
        gt_confs: classification targets (B, num_default)
        neg_ratio: negative / positive ratio
    Returns:
        conf_loss: classification loss
        loc_loss: regression loss
    """
    # loss: B x N
    # gt_confs: B x N
    pos_idx = gt_confs > 0
    num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.int32), axis=1)
    num_neg = num_pos * neg_ratio

    rank = tf.argsort(loss, axis=1, direction='DESCENDING')
    rank = tf.argsort(rank, axis=1)
    neg_idx = rank < tf.expand_dims(num_neg, 1)

    return pos_idx, neg_idx


class SSDLosses(object):
    """ Class for SSD Losses
    Attributes:
        neg_ratio: negative / positive ratio
        num_classes: number of classes
    """

    def __init__(self, neg_ratio, num_classes):
        self.neg_ratio = neg_ratio
        self.num_classes = num_classes

    def __call__(self, confs, locs, gt_confs, gt_locs):
        """ Compute losses for SSD
            regression loss: smooth L1
            classification loss: cross entropy
        Args:
            confs: outputs of classification heads (B, num_default, num_classes)
            locs: outputs of regression heads (B, num_default, 4)
            gt_confs: classification targets (B, num_default)
            gt_locs: regression targets (B, num_default, 4)
        Returns:
            conf_loss: classification loss
            loc_loss: regression loss
        """
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        # compute classification losses
        # without reduction
        temp_loss = cross_entropy(
            gt_confs, confs)

        pos_idx, neg_idx = hard_negative_mining(
            temp_loss, gt_confs, self.neg_ratio)

        # classification loss will consist of positive and negative examples

        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='sum')
        smooth_l1_loss = tf.keras.losses.Huber(reduction='sum')

        conf_loss = cross_entropy(
            gt_confs[tf.math.logical_or(pos_idx, neg_idx)],
            confs[tf.math.logical_or(pos_idx, neg_idx)])

        # regression loss only consist of positive examples
        loc_loss = smooth_l1_loss(
            # tf.boolean_mask(gt_locs, pos_idx),
            # tf.boolean_mask(locs, pos_idx))
            gt_locs[pos_idx],
            locs[pos_idx])

        num_pos = tf.reduce_sum(tf.dtypes.cast(pos_idx, tf.float32))

        conf_loss = conf_loss / num_pos
        loc_loss = loc_loss / num_pos

        return conf_loss, loc_loss


def create_losses(neg_ratio, num_classes):
    criterion = SSDLosses(neg_ratio, num_classes)

    return criterion



parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='../data')
parser.add_argument('--data-year', default='2007')
parser.add_argument('--arch', default='ssd300')
parser.add_argument('--num-examples', default=-1, type=int)
parser.add_argument('--pretrained-type', default='specified')
parser.add_argument('--checkpoint-dir', default='')
parser.add_argument('--checkpoint-path', default='')
parser.add_argument('--gpu-id', default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

NUM_CLASSES = 2
BATCH_SIZE = 1


def predict(imgs, default_boxes):
    confs, locs = ssd(imgs)

    confs = tf.squeeze(confs, 0)
    locs = tf.squeeze(locs, 0)

    confs = tf.math.softmax(confs, axis=-1)
    classes = tf.math.argmax(confs, axis=-1)
    scores = tf.math.reduce_max(confs, axis=-1)

    boxes = decode(default_boxes, locs)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, NUM_CLASSES):
        cls_scores = confs[:, c]

        score_idx = cls_scores > 0.6
        # cls_boxes = tf.boolean_mask(boxes, score_idx)
        # cls_scores = tf.boolean_mask(cls_scores, score_idx)
        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]

        nms_idx = calc_nms(cls_boxes, cls_scores, 0.45, 200)
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores


if __name__ == '__main__':
    with open('./config.yml') as f:
        cfg = yaml.load(f)

    try:
        config = cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))

    default_boxes = gen_default_boxes(config)

    batch_generator, info = create_batch_generator(
        args.data_dir, default_boxes,
        config['image_size'],
        BATCH_SIZE, args.num_examples, mode='test')

    try:
        ssd = create_ssd(NUM_CLASSES, args.arch,
                         args.pretrained_type,
                         args.checkpoint_dir,
                         args.checkpoint_path)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    os.makedirs('outputs/images', exist_ok=True)
    os.makedirs('outputs/detects', exist_ok=True)
    visualizer = ImageVisualizer(info['idx_to_name'], save_dir='outputs/images')

    for i, (filename, imgs, gt_confs, gt_locs) in enumerate(
        tqdm(batch_generator, total=info['length'],
             desc='Testing...', unit='images')):
        boxes, classes, scores = predict(imgs, default_boxes)
        filename = filename.numpy()[0].decode()
        original_image = Image.open(
            os.path.join(info['image_dir'], '{}.jpg'.format(filename)))
        boxes *= original_image.size * 2
        visualizer.save_image(
            original_image, boxes, classes, '{}.jpg'.format(filename))

        log_file = os.path.join('outputs/detects', '{}.txt')

        for cls, box, score in zip(classes, boxes, scores):
            cls_name = info['idx_to_name'][cls - 1]
            with open(log_file.format(cls_name), 'a') as f:
                f.write('{} {} {} {} {} {}\n'.format(
                    filename,
                    score,
                    *[coord for coord in box]))
