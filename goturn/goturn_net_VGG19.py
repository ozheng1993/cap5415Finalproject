import tensorflow as tf
import numpy as np
class TRACKNET: 
    def __init__(self, batch_size, train = True):
        self.parameters = {}
        self.batch_size = batch_size
        self.target = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
        self.image = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
        self.bbox = tf.placeholder(tf.float32, [batch_size, 4])
        self.train = train
        self.wd = 0.0005
    def build(self):
        ########### for target ###########
        # [filter_height, filter_width, in_channels, out_channels]
        self.target_conv1 = self._conv_relu_layer(bottom = self.target, filter_size = [3, 3, 3, 64],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_1")

        # now 224 x 224 x 64
        self.target_conv2 = self._conv_relu_layer(bottom = self.target_conv1, filter_size = [3, 3, 64, 64],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_2")

        # now 112 x 112 x 64
        self.target_pool1 = tf.nn.max_pool(self.target_conv2, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='target_pool1')
        
        # now 112 x 112 x 128
        self.target_lrn1 = tf.nn.local_response_normalization(self.target_pool1, depth_radius = 2, alpha=0.0001,
                                                    beta=0.75, name="target_lrn1")


        # now 112 x 112 x 128
        self.target_conv3 = self._conv_relu_layer(bottom = self.target_lrn1, filter_size = [3, 3, 64, 128],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_3")

        # now 112 x 112 x 128
        self.target_conv4 = self._conv_relu_layer(bottom = self.target_conv3, filter_size = [3, 3, 128, 128],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_4")

        # now 56 x 56 x 128
        self.target_pool2 = tf.nn.max_pool(self.target_conv4, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='target_pool2')
        

        # now 56 x 56 x 256
        self.target_lrn2 = tf.nn.local_response_normalization(self.target_pool2, depth_radius = 2, alpha=0.0001,
                                                    beta=0.75, name="target_lrn2")        

        # now 56 x 56 x 256
        self.target_conv5 = self._conv_relu_layer(bottom = self.target_lrn2, filter_size = [3, 3, 128, 256],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_5")

        # now 56 x 56 x 256
        self.target_conv6 = self._conv_relu_layer(bottom = self.target_conv5, filter_size = [3, 3, 256, 256],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_6")

        # now 56 x 56 x 256
        self.target_conv7 = self._conv_relu_layer(bottom = self.target_conv6, filter_size = [3, 3, 256, 256],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_7")

        # now 56 x 56 x 256
        self.target_conv8 = self._conv_relu_layer(bottom = self.target_conv7, filter_size = [3, 3, 256, 256],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_8")

        # now 28 x 28 x 256
        self.target_pool3 = tf.nn.max_pool(self.target_conv8, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='target_pool3')

        # now 28 x 28 x 512
        self.target_lrn3 = tf.nn.local_response_normalization(self.target_pool3, depth_radius = 2, alpha=0.0001,
                                                    beta=0.75, name="target_lrn3")  

        # now 28 x 28 x 512
        self.target_conv9 = self._conv_relu_layer(bottom = self.target_lrn3, filter_size = [3, 3, 256, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_9")

        # now 28 x 28 x 512
        self.target_conv10 = self._conv_relu_layer(bottom = self.target_conv9, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_10")

        # now 28 x 28 x 512
        self.target_conv11 = self._conv_relu_layer(bottom = self.target_conv10, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_11")
        
        # now 28 x 28 x 512
        self.target_conv12 = self._conv_relu_layer(bottom = self.target_conv11, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_12")        
      
        # now 14 x 14 x 512
        self.target_pool4 = tf.nn.max_pool(self.target_conv12, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='target_pool4')

        # now 14 x 14 x 512
        self.target_lrn4 = self._conv_relu_layer(bottom = self.target_pool4, filter_size = [3, 3, 256, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_lrn4")

        # now 14 x 14 x 512
        self.target_conv13 = self._conv_relu_layer(bottom = self.target_lrn4, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_13")

        # now 14 x 14 x 512
        self.target_conv14 = self._conv_relu_layer(bottom = self.target_conv13, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_14")

        # now 14 x 14 x 512
        self.target_conv15 = self._conv_relu_layer(bottom = self.target_conv14, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_15")

        # now 14 x 14 x 512
        self.target_conv16 = self._conv_relu_layer(bottom = self.target_conv15, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_conv_16")


        # now 7 x 7 x 512
        self.target_pool5 = tf.nn.max_pool(self.target_conv16, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='target_pool5')
        
        # now 7 x 7 x 512
        

        ########### for image ###########
        # [filter_height, filter_width, in_channels, out_channels]
        self.image_conv1 = self._conv_relu_layer(bottom = self.image, filter_size = [3, 3, 3, 64],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_1")

        # now 224 x 224 x 64
        self.image_conv2 = self._conv_relu_layer(bottom = self.image_conv1, filter_size = [3, 3, 64, 64],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_2")

        # now 112 x 112 x 64
        self.image_pool1 = tf.nn.max_pool(self.image_conv2, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='image_pool1')

        # now 112 x 112 x 128
        self.target_lrn1 = self._conv_relu_layer(bottom = self.target_pool1, filter_size = [3, 3, 256, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_lrn1")

        # now 112 x 112 x 128
        self.image_conv3 = self._conv_relu_layer(bottom = self.target_lrn1, filter_size = [3, 3, 64, 128],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_3")

        # now 112 x 112 x 128
        self.image_conv4 = self._conv_relu_layer(bottom = self.image_conv3, filter_size = [3, 3, 128, 128],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_4")

        # now 56 x 56 x 128
        self.image_pool2 = tf.nn.max_pool(self.image_conv4, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='image_pool2')

        # now 56 x 56 x 256
        self.target_lrn2 = self._conv_relu_layer(bottom = self.target_pool2, filter_size = [3, 3, 256, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_lrn2")


        # now 56 x 56 x 256
        self.image_conv5 = self._conv_relu_layer(bottom = self.target_lrn2, filter_size = [3, 3, 128, 256],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_5")

        # now 56 x 56 x 256
        self.image_conv6 = self._conv_relu_layer(bottom = self.image_conv5, filter_size = [3, 3, 256, 256],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_6")

        # now 56 x 56 x 256
        self.image_conv7 = self._conv_relu_layer(bottom = self.image_conv6, filter_size = [3, 3, 256, 256],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_7")

        # now 56 x 56 x 256
        self.image_conv8 = self._conv_relu_layer(bottom = self.image_conv7, filter_size = [3, 3, 256, 256],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_8")

        # now 28 x 28 x 256
        self.image_pool3 = tf.nn.max_pool(self.image_conv8, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='image_pool3')

        # now 28 x 28 x 512
        self.target_lrn3 = self._conv_relu_layer(bottom = self.target_pool3, filter_size = [3, 3, 256, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_lrn3")

        # now 28 x 28 x 512
        self.image_conv9 = self._conv_relu_layer(bottom = self.target_lrn3, filter_size = [3, 3, 256, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_9")

        # now 28 x 28 x 512
        self.image_conv10 = self._conv_relu_layer(bottom = self.image_conv9, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_10")

        # now 28 x 28 x 512
        self.image_conv11 = self._conv_relu_layer(bottom = self.image_conv10, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_11")
        
        # now 28 x 28 x 512
        self.image_conv12 = self._conv_relu_layer(bottom = self.image_conv11, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_12")        
      
        # now 14 x 14 x 512
        self.image_pool4 = tf.nn.max_pool(self.image_conv12, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='image_pool4')
        
        # now 14 x 14 x 512
        self.target_lrn4 = self._conv_relu_layer(bottom = self.target_pool4, filter_size = [3, 3, 256, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "target_lrn4")        

        # now 14 x 14 x 512
        self.image_conv13 = self._conv_relu_layer(bottom = self.target_lrn4, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_13")

        # now 14 x 14 x 512
        self.image_conv14 = self._conv_relu_layer(bottom = self.image_conv13, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_14")

        # now 14 x 14 x 512
        self.image_conv15 = self._conv_relu_layer(bottom = self.image_conv14, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_15")

        # now 14 x 14 x 512
        self.image_conv16 = self._conv_relu_layer(bottom = self.image_conv15, filter_size = [3, 3, 512, 512],
                                                    strides = [1,1,1,1], pad = 1, name = "image_conv_16")


        # now 7 x 7 x 512
        self.image_pool5 = tf.nn.max_pool(self.image_conv16, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                    padding='VALID', name='image_pool5')

        # now 7 x 7 x 512
        # tensorflow layer: n * w * h * c
        # but caffe layer is: n * c * h * w

        # tensorflow kernel: h * w * in_c * out_c
        # caffe kernel: out_c * in_c * h * w

        ########### Concatnate two layers ###########
        self.concat = tf.concat([self.target_pool5, self.image_pool5], axis = 3) # 0, 1, 2, 3 - > 2, 3, 1, 0

        # important, since caffe has different layer dimension order
        self.concat = tf.transpose(self.concat, perm=[0,3,1,2]) 

        ########### fully connencted layers ###########
        # 6 * 6 * 256 * 2 == 18432
        # assert self.fc1.get_shape().as_list()[1:] == [6, 6, 512]
        self.fc1 = self._fc_relu_layers(self.concat, dim = 4096, name = "fc1")
        if (self.train):
            self.fc1 = tf.nn.dropout(self.fc1, 0.5)


        self.fc2 = self._fc_relu_layers(self.fc1, dim = 4096, name = "fc2")
        if (self.train):
            self.fc2 = tf.nn.dropout(self.fc2, 0.5)

        self.fc3 = self._fc_relu_layers(self.fc2, dim = 4096, name = "fc3")
        if (self.train):
            self.fc3 = tf.nn.dropout(self.fc3, 0.5)

        self.fc4 = self._fc_layers(self.fc3, dim = 4, name = "fc4")

        self.print_shapes()
        self.loss = self._loss_layer(self.fc4, self.bbox ,name = "loss")
        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_weight_loss')
        self.loss_wdecay = self.loss + l2_loss

    def _loss_layer(self, bottom, label, name = None):
        diff = tf.subtract(self.fc4, self.bbox)
        diff_flat = tf.abs(tf.reshape(diff,[-1]))
        loss = tf.reduce_sum(diff_flat, name = name)
        return loss

    def _conv_relu_layer(self,bottom,filter_size, strides, pad = 0,bias_init = 0.0, group = 1, trainable = False, name = None):
        with tf.name_scope(name) as scope:

            if (pad > 0):
                paddings = [[0,0],[pad,pad],[pad,pad],[0,0]]
                bottom = tf.pad(bottom, paddings, "CONSTANT")
            kernel = tf.Variable(tf.truncated_normal(filter_size, dtype=tf.float32,
                                                     stddev=1e-2), trainable=trainable, name='weights')
            biases = tf.Variable(tf.constant(bias_init, shape=[filter_size[3]], dtype=tf.float32), trainable=trainable, name='biases')
            self.parameters[name] = [kernel, biases]
            if (group == 1):
                conv = tf.nn.conv2d(bottom, kernel, strides, padding='VALID')
                out = tf.nn.bias_add(conv, biases)
            elif (group == 2):
                kernel1, kernel2 = tf.split(kernel, num_or_size_splits=group, axis=3)
                bottom1, bottom2 = tf.split(bottom, num_or_size_splits=group, axis=3)
                conv1 = tf.nn.conv2d(bottom1, kernel1, strides, padding='VALID')
                conv2 = tf.nn.conv2d(bottom2, kernel2, strides, padding='VALID')
                conv = tf.concat([conv1, conv2], axis=3)
                out = tf.nn.bias_add(conv, biases)
            else:
                raise TypeError("number of groups not supported")

            # if not tf.get_variable_scope().reuse:
            #     weight_decay = tf.multiply(tf.nn.l2_loss(kernel), self.wd,
            #                            name='kernel_loss')
            #     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
            #                      weight_decay)


            out2 = tf.nn.relu(out, name=scope)
            _activation_summary(out2)
            out2 = tf.Print(out2, [tf.shape(out2)], message='Shape of %s' % name, first_n = 1, summarize=4)
            return out2

    def _fc_relu_layers(self, bottom, dim, name = None):
        with tf.name_scope(name) as scope:
            shape = int(np.prod(bottom.get_shape()[1:]))
            weights = tf.Variable(tf.truncated_normal([shape, dim],
                                    dtype=tf.float32, stddev=0.005), name='weights')
            bias = tf.Variable(tf.constant(1.0, shape=[dim], dtype=tf.float32), name='biases')
            bottom_flat = tf.reshape(bottom, [-1, shape])
            fc_weights = tf.nn.bias_add(tf.matmul(bottom_flat, weights), bias)
            self.parameters[name] = [weights, bias]


            if not tf.get_variable_scope().reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), self.wd,
                                       name='fc_relu_weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)



            top = tf.nn.relu(fc_weights, name=scope)
            _activation_summary(top)
            top = tf.Print(top, [tf.shape(top)], message='Shape of %s' % name, first_n = 1, summarize=4)
            return top

    def _fc_layers(self, bottom, dim, name = None):
        with tf.name_scope(name) as scope:
            shape = int(np.prod(bottom.get_shape()[1:]))
            weights = tf.Variable(tf.truncated_normal([shape, dim],
                                    dtype=tf.float32, stddev=0.005), name='weights')
            bias = tf.Variable(tf.constant(1.0, shape=[dim], dtype=tf.float32), name='biases')
            bottom_flat = tf.reshape(bottom, [-1, shape])
            top = tf.nn.bias_add(tf.matmul(bottom_flat, weights), bias, name=scope)
            self.parameters[name] = [weights, bias]

            if not tf.get_variable_scope().reuse:
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), self.wd,
                                       name='fc_weight_loss')
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 weight_decay)

            _activation_summary(top)
            top = tf.Print(top, [tf.shape(top)], message='Shape of %s' % name, first_n = 1, summarize=4)
            return top
    def _add_wd_and_summary(self, var, wd, collection_name=None):
        if collection_name is None:
            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var
    def print_shapes(self):
        print("%s:"%(self.image_conv1),self.image_conv1.get_shape().as_list())
        print("%s:"%(self.image_conv2),self.image_conv2.get_shape().as_list())        
        print("%s:"%(self.image_pool1),self.image_pool1.get_shape().as_list())
        print("%s:"%(self.image_lrn1),self.image_lrn1.get_shape().as_list())
        print("%s:"%(self.image_conv3),self.image_conv3.get_shape().as_list())
        print("%s:"%(self.image_conv4),self.image_conv4.get_shape().as_list())      
        print("%s:"%(self.image_pool2),self.image_pool2.get_shape().as_list())
        print("%s:"%(self.image_lrn2),self.image_lrn2.get_shape().as_list())
        print("%s:"%(self.image_conv5),self.image_conv5.get_shape().as_list())
        print("%s:"%(self.image_conv6),self.image_conv6.get_shape().as_list())            
        print("%s:"%(self.image_conv7),self.image_conv7.get_shape().as_list())
        print("%s:"%(self.image_conv8),self.image_conv8.get_shape().as_list())
        print("%s:"%(self.image_pool3),self.image_pool3.get_shape().as_list())
        print("%s:"%(self.image_lrn3),self.image_lrn3.get_shape().as_list())        
        print("%s:"%(self.image_conv9),self.image_conv9.get_shape().as_list())
        print("%s:"%(self.image_conv10),self.image_conv10.get_shape().as_list())            
        print("%s:"%(self.image_conv11),self.image_conv11.get_shape().as_list())
        print("%s:"%(self.image_conv12),self.image_conv12.get_shape().as_list())        
        print("%s:"%(self.image_pool4),self.image_pool4.get_shape().as_list())       
        print("%s:"%(self.image_lrn4),self.image_lrn4.get_shape().as_list())         
        print("%s:"%(self.image_conv13),self.image_conv13.get_shape().as_list())            
        print("%s:"%(self.image_conv14),self.image_conv14.get_shape().as_list())
        print("%s:"%(self.image_conv15),self.image_conv15.get_shape().as_list())        
        print("%s:"%(self.image_conv16),self.image_conv16.get_shape().as_list())         
        print("%s:"%(self.image_pool5),self.image_pool5.get_shape().as_list())                 
        print("%s:"%(self.concat),self.concat.get_shape().as_list())
        print("%s:"%(self.fc1),self.fc1.get_shape().as_list())
        print("%s:"%(self.fc2),self.fc2.get_shape().as_list())
        print("%s:"%(self.fc3),self.fc3.get_shape().as_list())
        print("%s:"%(self.fc4),self.fc4.get_shape().as_list())
        print("kernel_sizes:")
        for key in self.parameters:
            print("%s:"%(key),self.parameters[key][0].get_shape().as_list())

    def load_weight_from_dict(self,weights_dict,sess):
        # for convolutional layers
        sess.run(self.parameters['target_conv_1'][0].assign(weights_dict['conv1']['weights']))
        sess.run(self.parameters['target_conv_2'][0].assign(weights_dict['conv2']['weights']))
        sess.run(self.parameters['target_conv_3'][0].assign(weights_dict['conv3']['weights']))
        sess.run(self.parameters['target_conv_4'][0].assign(weights_dict['conv4']['weights']))
        sess.run(self.parameters['target_conv_5'][0].assign(weights_dict['conv5']['weights']))
        sess.run(self.parameters['target_conv_6'][0].assign(weights_dict['conv6']['weights']))
        sess.run(self.parameters['target_conv_7'][0].assign(weights_dict['conv7']['weights']))
        sess.run(self.parameters['target_conv_8'][0].assign(weights_dict['conv8']['weights']))
        sess.run(self.parameters['target_conv_9'][0].assign(weights_dict['conv9']['weights']))
        sess.run(self.parameters['target_conv_10'][0].assign(weights_dict['conv10']['weights']))
        sess.run(self.parameters['target_conv_11'][0].assign(weights_dict['conv11']['weights']))
        sess.run(self.parameters['target_conv_12'][0].assign(weights_dict['conv12']['weights']))
        sess.run(self.parameters['target_conv_13'][0].assign(weights_dict['conv13']['weights']))
        sess.run(self.parameters['target_conv_14'][0].assign(weights_dict['conv14']['weights']))
        sess.run(self.parameters['target_conv_15'][0].assign(weights_dict['conv15']['weights']))
        sess.run(self.parameters['target_conv_16'][0].assign(weights_dict['conv16']['weights']))
        
        sess.run(self.parameters['image_conv_1'][0].assign(weights_dict['conv1_p']['weights']))
        sess.run(self.parameters['image_conv_2'][0].assign(weights_dict['conv2_p']['weights']))
        sess.run(self.parameters['image_conv_3'][0].assign(weights_dict['conv3_p']['weights']))
        sess.run(self.parameters['image_conv_4'][0].assign(weights_dict['conv4_p']['weights']))
        sess.run(self.parameters['image_conv_5'][0].assign(weights_dict['conv5_p']['weights']))
        sess.run(self.parameters['image_conv_6'][0].assign(weights_dict['conv6_p']['weights']))
        sess.run(self.parameters['image_conv_7'][0].assign(weights_dict['conv7_p']['weights']))
        sess.run(self.parameters['image_conv_8'][0].assign(weights_dict['conv8_p']['weights']))
        sess.run(self.parameters['image_conv_9'][0].assign(weights_dict['conv9_p']['weights']))
        sess.run(self.parameters['image_conv_10'][0].assign(weights_dict['conv10_p']['weights']))
        sess.run(self.parameters['image_conv_11'][0].assign(weights_dict['conv11_p']['weights']))
        sess.run(self.parameters['image_conv_12'][0].assign(weights_dict['conv12_p']['weights']))
        sess.run(self.parameters['image_conv_13'][0].assign(weights_dict['conv13_p']['weights']))
        sess.run(self.parameters['image_conv_14'][0].assign(weights_dict['conv14_p']['weights']))
        sess.run(self.parameters['image_conv_15'][0].assign(weights_dict['conv15_p']['weights']))
        sess.run(self.parameters['image_conv_16'][0].assign(weights_dict['conv16_p']['weights']))        

        sess.run(self.parameters['target_conv_1'][1].assign(weights_dict['conv1']['bias']))
        sess.run(self.parameters['target_conv_2'][1].assign(weights_dict['conv2']['bias']))
        sess.run(self.parameters['target_conv_3'][1].assign(weights_dict['conv3']['bias']))
        sess.run(self.parameters['target_conv_4'][1].assign(weights_dict['conv4']['bias']))
        sess.run(self.parameters['target_conv_5'][1].assign(weights_dict['conv5']['bias']))
        sess.run(self.parameters['target_conv_6'][1].assign(weights_dict['conv6']['bias']))
        sess.run(self.parameters['target_conv_7'][1].assign(weights_dict['conv7']['bias']))
        sess.run(self.parameters['target_conv_8'][1].assign(weights_dict['conv8']['bias']))
        sess.run(self.parameters['target_conv_9'][1].assign(weights_dict['conv9']['bias']))
        sess.run(self.parameters['target_conv_10'][1].assign(weights_dict['conv10']['bias']))        
        sess.run(self.parameters['target_conv_11'][1].assign(weights_dict['conv11']['bias']))
        sess.run(self.parameters['target_conv_12'][1].assign(weights_dict['conv12']['bias']))
        sess.run(self.parameters['target_conv_13'][1].assign(weights_dict['conv13']['bias']))
        sess.run(self.parameters['target_conv_14'][1].assign(weights_dict['conv14']['bias']))
        sess.run(self.parameters['target_conv_15'][1].assign(weights_dict['conv15']['bias']))        
        sess.run(self.parameters['target_conv_16'][1].assign(weights_dict['conv16']['bias']))            
        
        sess.run(self.parameters['image_conv_1'][1].assign(weights_dict['conv1_p']['bias']))
        sess.run(self.parameters['image_conv_2'][1].assign(weights_dict['conv2_p']['bias']))
        sess.run(self.parameters['image_conv_3'][1].assign(weights_dict['conv3_p']['bias']))
        sess.run(self.parameters['image_conv_4'][1].assign(weights_dict['conv4_p']['bias']))
        sess.run(self.parameters['image_conv_5'][1].assign(weights_dict['conv5_p']['bias']))
        sess.run(self.parameters['image_conv_6'][1].assign(weights_dict['conv6_p']['bias']))
        sess.run(self.parameters['image_conv_7'][1].assign(weights_dict['conv7_p']['bias']))
        sess.run(self.parameters['image_conv_8'][1].assign(weights_dict['conv8_p']['bias']))
        sess.run(self.parameters['image_conv_9'][1].assign(weights_dict['conv9_p']['bias']))
        sess.run(self.parameters['image_conv_10'][1].assign(weights_dict['conv10_p']['bias']))
        sess.run(self.parameters['image_conv_11'][1].assign(weights_dict['conv11_p']['bias']))
        sess.run(self.parameters['image_conv_12'][1].assign(weights_dict['conv12_p']['bias']))
        sess.run(self.parameters['image_conv_13'][1].assign(weights_dict['conv13_p']['bias']))
        sess.run(self.parameters['image_conv_14'][1].assign(weights_dict['conv14_p']['bias']))
        sess.run(self.parameters['image_conv_15'][1].assign(weights_dict['conv15_p']['bias']))        
        sess.run(self.parameters['image_conv_16'][1].assign(weights_dict['conv16_p']['bias'])) 


        # for fully connected layers
        sess.run(self.parameters['fc1'][0].assign(weights_dict['fc6-new']['weights']))
        sess.run(self.parameters['fc2'][0].assign(weights_dict['fc7-new']['weights']))
        sess.run(self.parameters['fc3'][0].assign(weights_dict['fc7-newb']['weights']))
        sess.run(self.parameters['fc4'][0].assign(weights_dict['fc8-shapes']['weights']))

        sess.run(self.parameters['fc1'][1].assign(weights_dict['fc6-new']['bias']))
        sess.run(self.parameters['fc2'][1].assign(weights_dict['fc7-new']['bias']))
        sess.run(self.parameters['fc3'][1].assign(weights_dict['fc7-newb']['bias']))
        sess.run(self.parameters['fc4'][1].assign(weights_dict['fc8-shapes']['bias']))


    
    def test(self):
        sess = tf.Session()
        a = np.full((self.batch_size,227,227,3), 1) # numpy.full(shape, fill_value, dtype=None, order='C')
        b = np.full((self.batch_size,227,227,3), 2)
        sess.run(tf.global_variables_initializer())

        sess.run([self.fc4],feed_dict={self.image:a, self.target:b})





def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.debug("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)

if __name__ == "__main__":
    tracknet = TRACKNET(10)
    tracknet.build()
    sess = tf.Session()
    a = np.full((tracknet.batch_size,227,227,3), 1)
    b = np.full((tracknet.batch_size,227,227,3), 2)
    sess.run(tf.global_variables_initializer())
    sess.run([tracknet.image_pool5],feed_dict={tracknet.image:a, tracknet.target:b})



