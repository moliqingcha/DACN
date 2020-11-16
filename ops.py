import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def conv2d(inputs, rate_field, num_outputs, kernel_size, scope, stride=1, rate=1, 
            is_train=True, bias=True, norm=True, activation=True, d_format='NHWC'):
    
  
    ### bias ###
    if bias:
        outputs = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride=stride,
               data_format=d_format, rate=rate, activation_fn=None, scope=scope)
    else:
        outputs = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride=stride,
               data_format=d_format, rate=rate, activation_fn=None, biases_initializer=None, scope=scope)
    
    ### BN ###
    if norm:
        outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=None,
               epsilon=1e-5, is_training=is_train, scope=scope+'/batch_norm', data_format=d_format)
  
    if activation:
        outputs = tf.nn.relu(outputs, name=scope+'/relu')
    
    return outputs
#———————————————————————————————————————————————————————————#  

#######################################################################################################################
def bn(inputs, scope, is_train=True, d_format='NHWC'):
    
    outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=None,
           epsilon=1e-5, is_training=is_train, scope='nonlocal/batch_norm', data_format=d_format)
   
    return outputs

def _relu(inputs, scope):
    
    outputs = tf.nn.relu(inputs, name=scope+'/relu')
   
    return outputs
 
def _tanh(inputs, scope):
    
    outputs = tf.nn.tanh(inputs, name=scope+'/tanh')
   
    return outputs

def _leaky_relu(inputs, scope):
    
    outputs = tf.nn.leaky_relu(inputs, name=scope+'/leaky_relu')
   
    return outputs
 
def _sigmoid(inputs, scope):
    
    outputs = tf.nn.sigmoid(inputs, name=scope+'/sigmoid')
   
    return outputs

def _max_pool2d(inputs, kernel_size, scope, stride=2, padding='SAME', data_format='NHWC'):
    
    outputs = tf.contrib.layers.max_pool2d(inputs, kernel_size, stride=stride, 
           scope=scope+'/max_pool', padding=padding, data_format=data_format)
    
    return outputs

def _avg_pool2d(inputs, kernel_size, scope, stride=2, padding='SAME', data_format='NHWC'):
    
    outputs = tf.contrib.layers.avg_pool2d(inputs, kernel_size, stride=stride, 
           scope=scope+'/avg_pool', padding=padding, data_format=data_format)
    
    return outputs

def deconv(inputs, num_outputs, kernel_size, scope, new_height=None, new_width=None, stride=2, is_train=True, d_format='NHWC'):
    
    stride_new = [stride, stride]      
    outputs = tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size, scope=scope+'/deconv', stride=stride_new,
           padding='SAME', data_format=d_format, activation_fn=None, biases_initializer=None)
    
    return outputs

def bilinear(inputs, num_outputs, kernel_size, scope, new_height=None, new_width=None, stride=2, is_train=True, d_format='NHWC'):
    
    size_new = (new_height,new_width)    
    outputs = tf.image.resize_bilinear(inputs, size=size_new, align_corners=True, name=scope+'/bilinear')
    
    return outputs

def deconv_unit(inputs, num_outputs, kernel_size, scope, new_height=None, new_width=None, stride=2, is_train=True, d_format='NHWC'):
    
    stride_new = [stride, stride]      
    outputs = tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size, scope=scope+'/deconv', stride=stride_new,
           padding='SAME', data_format=d_format, activation_fn=None, biases_initializer=None)
    
    outputs = tf.contrib.layers.batch_norm(outputs, decay=0.9, center=True, scale=True, activation_fn=tf.nn.relu,
               epsilon=1e-5, is_training=is_train, scope=scope+'/batch_norm', data_format=d_format)
    
    return outputs
