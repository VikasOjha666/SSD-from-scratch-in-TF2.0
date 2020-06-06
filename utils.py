import itertools
import math
import tensorflow as tf


def gen_default_boxes(config):
    """
    This function generates the default boxes for the feature maps
    according to the number of feature maps and the aspect ratio specified for
    the boxes.

    Args:
     config: Contains the various information about the boxes.
        scales: size of boxes relative to image's size.
        fm_size:sizes of feature maps
        ratio:aspect ratios for various boxes.

    Returns:
        default_boxes:tensor of shape (num_default,4)
        with format(cx,cy,w,h)

    """

    default_boxes=[]
    scales=config['scales']
    feat_map_size=config['feat_map_size']
    aspect_ratios=config['aspect_ratios']

    for m,fm_size in enumerate(feat_map_size):
        for i,j in itertools.product(range(fm_size),repeat=2):
            cx=(j+0.5)/fm_size
            cy=(i+0.5)/fm_size

            default_boxes.append([cx,cy,scales[m],scales[m]])

            default_boxes.append([cx,cy,math.sqrt(scales[m]*scales[m+1]),
                                math.sqrt(scales[m]*scales[m+1])])
            for ar in aspect_ratios[m]:
                r=math.sqrt(ar)
                default_boxes.append([cx,cy,scales[m]*r,scales[m]/r])

                default_boxes.append([cx,cy,scales[m] / r,scales[m] * r])
    default_boxes=tf.constant(default_boxes)
    default_boxes=tf.clip_by_value(default_boxes,0.0,1.0)

    return default_boxes


def calc_area(top_left,bottom_right):
    """
    It calculates the area of the bounding box when its top_left and bottom_left coordinates are passed.

    Args:
        top_left: tensor (num_boxes,2)
        bottom_right: tensor (num_boxes,2)
    Returns:
        area:tensor (num_boxes,)
    """

    #top_left:(x1,y1)
    #bottom_right:(x2,y2)

    HW=tf.clip_by_value(bottom_right-top_left,0.0,512.0) #x1-x2=width and y2-y1=height.
    area=HW[...,0] * HW[...,1]  #W X H.The ... operator is used for specifying that rows can have any shape its same as tf None.
    return area

def calc_iou(boxes_a,boxes_b):
    """
    This function calculates IOU of two boxes.

    Args:
        boxes_a: tensor(num_boxes,4)
        boxes_b: tensor(num_boxes,4)
    Returns:
        iou scores between the boxes boxes_a and boxes_b
    """

    #boxes_a shape=num_boxes,1,4
    boxes_a=tf.expand_dims(boxes_a,1)

    #boxes_b shape=1,num_boxes_b,4
    boxes_b=tf.expand_dims(boxes_b,0)

    #These forms the coordinates of the overlapping areas.
    top_left=tf.math.maximum(boxes_a[...,:2],boxes_b[...,:2])
    bottom_right=tf.math.minimum(boxes_a[...,2:],boxes_b[...,2:])

    overlap_area=calc_area(top_left,bottom_right)
    area_a=calc_area(boxes_a[...,:2],boxes_a[...,2:])
    area_b=calc_area(boxes_b[...,:2],boxes_b[...,2:])

    overlap=overlap_area/(area_a+area_b-overlap_area)
    return overlap

def calc_target(default_boxes,gt_boxes,gt_labels,iou_threshold=0.5):
    """
    This function calculates regression and classification targets.
    i.e.As the boxes which have good IOU are only positive targets
    and the optimization has to be done on them.

    Args:
        default_boxes: tensor(num_default,4) of format(cx,cy,w,h)
        gt_boxes: tensor(num_gt,4) of format(xmin,ymin,xmax,ymax)
        gt_labels: tensor(num_gt,)

    Returns:
        gt_confs:classification target,tensor (num_default,)
        gt_locs:regression targets,tensor (num_default,4)

    """
    #This converts the default boxes to (xmin,ymin,xmax,ymax) to match with ground truth boxes.
    #As the ground truth boxes are in form (xmin,ymin,xmax,ymax).
    transformed_default_boxes=transform_center_to_corner(default_boxes)

    iou=calc_iou(transformed_default_boxes,gt_boxes)

    best_gt_iou=tf.math.reduce_max(iou,1)
    best_gt_idx=tf.math.argmax(iou,1)

    best_default_iou = tf.math.reduce_max(iou, 0)
    best_default_idx = tf.math.argmax(iou, 0)

    #Get the index of the gt box with which we had maximum overlap(IOU scores.)
    best_gt_idx = tf.tensor_scatter_nd_update(best_gt_idx,tf.expand_dims(best_default_idx, 1),tf.range(best_default_idx.shape[0], dtype=tf.int64))

    best_gt_iou = tf.tensor_scatter_nd_update(best_gt_iou,tf.expand_dims(best_default_idx, 1),tf.ones_like(best_default_idx, dtype=tf.float32))

    gt_confs=tf.gather(gt_labels,best_gt_idx)
    gt_confs=tf.where(tf.less(best_gt_iou,iou_threshold),tf.zeros_like(gt_confs),gt_confs)
    gt_boxes=tf.gather(gt_boxes,best_gt_idx)
    gt_locs=encode(default_boxes,gt_boxes)

    return gt_confs,gt_locs

def encode(default_boxes,boxes,variance=[0.1,0.2]):
    """
    Compute regression value i.e the value regression value on which we want to
    optimize the network.

    Args:
        default_boxes: tensor(num_default,4)
        of format (cx,cy,w,h)

        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
        variance: variance for center point and size
    Returns:
        locs: regression values, tensor (num_default, 4)

    """
    #This converts the default boxes to (xmin,ymin,xmax,ymax)
    transformed_boxes=transform_corner_to_center(boxes)
    locs = tf.concat([
        (transformed_boxes[..., :2] - default_boxes[:, :2]
         ) / (default_boxes[:, 2:] * variance[0]),
        tf.math.log(transformed_boxes[..., 2:] / default_boxes[:, 2:]) / variance[1]],
        axis=-1)

    return locs

def decode(default_boxes, locs, variance=[0.1, 0.2]):
    """ Decode regression values back to coordinates
    Args:
        default_boxes: tensor (num_default, 4)
                       of format (cx, cy, w, h)
        locs: tensor (batch_size, num_default, 4)
              of format (cx, cy, w, h)
        variance: variance for center point and size
    Returns:
        boxes: tensor (num_default, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    locs = tf.concat([
        locs[..., :2] * variance[0] *
        default_boxes[:, 2:] + default_boxes[:, :2],
        tf.math.exp(locs[..., 2:] * variance[1]) * default_boxes[:, 2:]], axis=-1)

    boxes = transform_center_to_corner(locs)

    return boxes

def transform_corner_to_center(boxes):
    """
    Transform the boxes coordinates from (xmin,ymin,xmax,ymax)
    to format (cx,cy,w,h)

    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    """

    center_box = tf.concat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]], axis=-1)

    return center_box

def transform_center_to_corner(boxes):
    """ Transform boxes of format (cx, cy, w, h)
        to format (xmin, ymin, xmax, ymax)
    Args:
        boxes: tensor (num_boxes, 4)
               of format (cx, cy, w, h)
    Returns:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
    """
    corner_box = tf.concat([
        boxes[..., :2] - boxes[..., 2:] / 2,
        boxes[..., :2] + boxes[..., 2:] / 2], axis=-1)

    return corner_box


def calc_nms(boxes, scores, nms_threshold, limit=200):
    """ Perform Non Maximum Suppression algorithm
        to eliminate boxes with high overlap

    Args:
        boxes: tensor (num_boxes, 4)
               of format (xmin, ymin, xmax, ymax)
        scores: tensor (num_boxes,)
        nms_threshold: NMS threshold
        limit: maximum number of boxes to keep

    Returns:
        idx: indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return tf.constant([], dtype=tf.int32)
    selected = [0]
    idx = tf.argsort(scores, direction='DESCENDING')
    idx = idx[:limit]
    boxes = tf.gather(boxes, idx)

    iou = calc_iou(boxes, boxes)

    while True:
        row = iou[selected[-1]]
        next_indices = row <= nms_threshold
        # iou[:, ~next_indices] = 1.0
        iou = tf.where(
            tf.expand_dims(tf.math.logical_not(next_indices), 0),
            tf.ones_like(iou, dtype=tf.float32),
            iou)

        if not tf.math.reduce_any(next_indices):
            break

        selected.append(tf.argsort(
            tf.dtypes.cast(next_indices, tf.int32), direction='DESCENDING')[0].numpy())

    return tf.gather(idx, selected)
