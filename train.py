import numpy as np
import tensorflow as tf
import os
import argparse
import time
from SSD import create_ssd
from utils import gen_default_boxes
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import sys
import yaml
from data_utils import create_batch_generator


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
parser.add_argument('--data-dir', default='../dataset')
parser.add_argument('--data-year', default='2007')
parser.add_argument('--arch', default='ssd300')
parser.add_argument('--batch-size', default=2, type=int)
parser.add_argument('--num-batches', default=-1, type=int)
parser.add_argument('--neg-ratio', default=3, type=int)
parser.add_argument('--initial-lr', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--num-epochs', default=120, type=int)
parser.add_argument('--checkpoint-dir', default='checkpoints')
parser.add_argument('--pretrained-type', default='base')
parser.add_argument('--gpu-id', default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

NUM_CLASSES = 2

@tf.function
def train_step(imgs,gt_confs,gt_locs,ssd,criterion,optimizer):
    with tf.GradientTape() as tape:
        confs,locs=ssd(imgs)

        conf_loss,loc_loss=criterion(confs,locs,gt_confs,gt_locs)
        loss=conf_loss+loc_loss
        l2_loss=[tf.nn.l2_loss(t) for t in ssd.trainable_variables]
        l2_loss=args.weight_decay*tf.math.reduce_sum(l2_loss)
        loss+=l2_loss
    gradients=tape.gradient(loss,ssd.trainable_variables)
    optimizer.apply_gradients(zip(gradients,ssd.trainable_variables))
    return loss,conf_loss,loc_loss,l2_loss

if __name__=='__main__':
    os.makedirs(args.checkpoint_dir,exist_ok=True)

    with open('./config.yml') as f:
        cfg=yaml.load(f)

    try:
        config=cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))

    default_boxes=gen_default_boxes(config)

    batch_generator,val_generator,info=create_batch_generator(
    args.data_dir,default_boxes,config['image_size'],
    args.batch_size,args.num_batches,
    mode='train',augmentation=['flip']
    )

    try:
        ssd=create_ssd(NUM_CLASSES,args.arch,
                        args.pretrained_type,
                        checkpoint_dir=args.checkpoint_dir)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    criterion=create_losses(args.neg_ratio,NUM_CLASSES)
    steps_per_epoch=info['length']//args.batch_size

    lr_fn = PiecewiseConstantDecay(
        boundaries=[int(steps_per_epoch * args.num_epochs * 2 / 3),
                    int(steps_per_epoch * args.num_epochs * 5 / 6)],
        values=[args.initial_lr, args.initial_lr * 0.1, args.initial_lr * 0.01])

    optimizer=tf.keras.optimizers.SGD(
    learning_rate=lr_fn,
    momentum=args.momentum
    )

    train_log_dir='logs/train'
    val_log_dir='logs/val'
    train_summary_writer=tf.summary.create_file_writer(train_log_dir)
    val_summary_writer=tf.summary.create_file_writer(val_log_dir)

    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        start = time.time()
        for i, (_,imgs, gt_confs, gt_locs) in enumerate(batch_generator):
            loss, conf_loss, loc_loss, l2_loss = train_step(
                imgs, gt_confs, gt_locs, ssd, criterion, optimizer)
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
            if (i + 1) % 50 == 0:
                print('Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                    epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss))

        avg_val_loss = 0.0
        avg_val_conf_loss = 0.0
        avg_val_loc_loss = 0.0
        for i, (_,imgs, gt_confs, gt_locs) in enumerate(val_generator):
            val_confs, val_locs = ssd(imgs)
            val_conf_loss, val_loc_loss = criterion(
                val_confs, val_locs, gt_confs, gt_locs)
            val_loss = val_conf_loss + val_loc_loss
            avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
            avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
            avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', avg_val_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_val_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_val_loc_loss, step=epoch)

        if (epoch + 1) % 10 == 0:
            ssd.save_weights(
                os.path.join(args.checkpoint_dir, 'ssd_epoch_{}.h5'.format(epoch + 1)))
