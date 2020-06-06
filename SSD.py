import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
import numpy as np
import os



def create_vgg16_layers():
    vgg16_conv4 = [
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.MaxPool2D(2, 2, padding='same'),

        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
    ]

    x = layers.Input(shape=[None, None, 3])
    out = x
    for layer in vgg16_conv4:
        out = layer(out)

    vgg16_conv4 = tf.keras.Model(x, out)

    vgg16_conv7 = [
        # Difference from original VGG16:
        # 5th maxpool layer has kernel size = 3 and stride = 1
        layers.MaxPool2D(3, 1, padding='same'),
        # atrous conv2d for 6th block
        layers.Conv2D(1024, 3, padding='same',
                      dilation_rate=6, activation='relu'),
        layers.Conv2D(1024, 1, padding='same', activation='relu'),
    ]

    x = layers.Input(shape=[None, None, 512])
    out = x
    for layer in vgg16_conv7:
        out = layer(out)

    vgg16_conv7 = tf.keras.Model(x, out)

    return vgg16_conv4, vgg16_conv7


def create_extra_layers():
    """ Create extra layers
        8th to 11th blocks
    """
    extra_layers = [
        # 8th block output shape: B, 512, 10, 10
        Sequential([
            layers.Conv2D(256, 1, activation='relu'),
            layers.Conv2D(512, 3, strides=2, padding='same',
                          activation='relu'),
        ]),
        # 9th block output shape: B, 256, 5, 5
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, strides=2, padding='same',
                          activation='relu'),
        ]),
        # 10th block output shape: B, 256, 3, 3
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, activation='relu'),
        ]),
        # 11th block output shape: B, 256, 1, 1
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, activation='relu'),
        ]),
        # 12th block output shape: B, 256, 1, 1
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 4, activation='relu'),
        ])
    ]

    return extra_layers


def create_conf_head_layers(num_classes):
    """ Create layers for classification
    """
    conf_head_layers = [
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 4th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 7th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 8th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 9th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 10th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 11th block
        layers.Conv2D(4 * num_classes, kernel_size=1)  # for 12th block
    ]

    return conf_head_layers


def create_loc_head_layers():
    """ Create layers for regression
    """
    loc_head_layers = [
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=1)
    ]

    return loc_head_layers




class SSD(Model):

    def __init__(self,num_classes,arch='ssd300'):
        """
        It initializes different head of the network as the network is branched into
        two parts one for conf score and one to predict bounding boxes.
        """
        super(SSD,self).__init__()
        self.num_classes=num_classes
        self.vgg16_conv4,self.vgg16_conv7=create_vgg16_layers()
        self.batch_norm=layers.BatchNormalization(beta_initializer='glorot_uniform',
                                                gamma_initializer='glorot_uniform')
        self.extra_layers=create_extra_layers()
        self.conf_head_layers=create_conf_head_layers(num_classes)
        self.loc_head_layers=create_loc_head_layers()

        if arch=='ssd300':
            self.extra_layers.pop(-1)
            self.conf_head_layers.pop(-2)
            self.loc_head_layers.pop(-2)
    def calc_head(self,x,idx):
        """
        Args:
            x: the input feature map
            idx: index of the head layer
        Returns:
            conf: output of the idx-th classification head
            loc: output of the idx-th regression head
        """

        """
        This function returns the output of classification and
        regression head based on the index supplied.This
        has been done so that we can obtain the classification of
        different feature maps as different feature map features helps
        in detecting objects of different sizes.
        """

        conf=self.conf_head_layers[idx](x)
        conf=tf.reshape(conf,[conf.shape[0],-1,self.num_classes])

        loc=self.loc_head_layers[idx](x)
        loc=tf.reshape(loc,[loc.shape[0],-1,4])

        return conf,loc

    def init_vgg16(self):
        """
        Initialize the VGG16 layers and rest.The VGG layers will be
        initialized with pretrained weights provided by keras.applications
        while rest will use weights from xavier initialization.
        """

        original_vgg=VGG16(weights='imagenet')
        for i in range(len(self.vgg16_conv4.layers)):
            self.vgg16_conv4.get_layer(index=i).set_weights(
            original_vgg.get_layer(index=i).get_weights()
            )

        fc1_weights,fc1_biases=original_vgg.get_layer(index=-3).get_weights()
        fc2_weights,fc2_biases=original_vgg.get_layer(index=-2).get_weights()

        conv6_weights = np.random.choice(
            np.reshape(fc1_weights, (-1,)), (3, 3, 512, 1024))
        conv6_biases = np.random.choice(
            fc1_biases, (1024,))

        conv7_weights = np.random.choice(
            np.reshape(fc2_weights, (-1,)), (1, 1, 1024, 1024))
        conv7_biases = np.random.choice(
            fc2_biases, (1024,))

        self.vgg16_conv7.get_layer(index=2).set_weights(
            [conv6_weights, conv6_biases])
        self.vgg16_conv7.get_layer(index=3).set_weights(
            [conv7_weights, conv7_biases])

    def call(self,x):
        """ The forward pass
        Args:
            x: the input image
        Returns:
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """

        confs=[]
        locs=[]
        head_idx=0
        for i in range(len(self.vgg16_conv4.layers)):
            x=self.vgg16_conv4.get_layer(index=i)(x)
            if i==len(self.vgg16_conv4.layers)-5:
                conf,loc=self.calc_head(self.batch_norm(x),head_idx)
                confs.append(conf)
                locs.append(loc)
                head_idx+=1
        x=self.vgg16_conv7(x)
        conf,loc=self.calc_head(x,head_idx)
        confs.append(conf)
        locs.append(loc)
        head_idx+=1

        for layer in self.extra_layers:
            x=layer(x)
            conf,loc=self.calc_head(x,head_idx)
            confs.append(conf)
            locs.append(loc)
            head_idx+=1
        confs=tf.concat(confs,axis=1)
        locs=tf.concat(locs,axis=1)

        return confs,locs

def create_ssd(num_classes,arch,pretrained_type,checkpoint_dir=None,checkpoint_path=None):
    """ Create SSD model and load pretrained weights
    Args:
        num_classes: number of classes
        pretrained_type: type of pretrained weights, can be either 'VGG16' or 'ssd'
        weight_path: path to pretrained weights
    Returns:
        net: the SSD model
    """

    net=SSD(num_classes,arch)
    net(tf.random.normal((1,512,512,3)))
    if pretrained_type=='base':
        net.init_vgg16()
    elif pretrained_type=='latest':
        try:
            paths=[os.path.join(checkpoint_dir,path) for path in os.listdir(checkpoint_dir)]
            latest=sorted(paths,key=os.path.getmtime)[-1]
            net.load_weights()

        except AttributeError as e:
            print('Please make sure there is at least one checkpoint at {}'.format(
                checkpoint_dir))
            print('The model will be loaded from base weights.')
            net.init_vgg16()
        except ValueError as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    latest, arch))
        except Exception as e:
            print(e)
            raise ValueError('Please check if checkpoint_dir is specified')

    elif pretrained_type == 'specified':
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                'Not a valid checkpoint file: {}'.format(checkpoint_path))

        try:
            net.load_weights(checkpoint_path)
        except Exception as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    checkpoint_path, arch))
    else:
        raise ValueError('Unknown pretrained type: {}'.format(pretrained_type))
    return net
