import random
random.seed(7)
import os
import numpy as np
import tensorflow as tf
from data_reader import H5DataLoader
from img_utils import imsave
from denseunet import DenseUnet
from acmdenseunet import AcmDenseUnet
from ccv import CCV
import ops

class Actions(object):
#—————————————————————————————————————————————————————#
    def __init__(self, sess, conf):
        
        #——————————————  step：1  ——————————————#
       
        self.sess = sess
        self.conf = conf
        self.conv_size = (3, 3)
        self.pool_size = (2, 2)
        self.data_format = 'NHWC'
        self.axis, self.channel_axis, self.batch_axis = (1, 2), 3, 0
        
        self.input_shape = [conf.batchsize, conf.height, conf.width, conf.channel]
        self.output_shape = [conf.batchsize, conf.height, conf.width]
        
        
        #——————————————  step：2  ——————————————#
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sample_dir):
            os.makedirs(conf.sample_dir)
            
        #——————————————  step：3  ——————————————#
        if self.conf.gpu_num==1:
            self.configure_networks_single()
        else:
            self.configure_networks_multi()
#———————————————————————————— configure_networks_single —————————————————————————# 
    def configure_networks_single(self):
        
        #——————————————  step：1  ——————————————#
        self.inputs = tf.placeholder(tf.float32, self.input_shape, name='inputs')
        self.annotations = tf.placeholder(tf.int64, self.output_shape, name='annotations')
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        expand_annotations = tf.expand_dims(self.annotations, -1, name='annotations/expand_dims')
        one_hot_annotations = tf.squeeze(expand_annotations, axis=[self.channel_axis],name='annotations/squeeze')
        one_hot_annotations = tf.one_hot(one_hot_annotations, depth=self.conf.class_num,
            axis=self.channel_axis, name='annotations/one_hot')
       
        #——————————————  step：2  ——————————————# 
        
        if self.conf.network_name=="denseunet":
            model = DenseUnet(self.sess, self.conf, self.is_train)
            self.outputs, self.rates = model.inference(self.inputs)
        if self.conf.network_name=="acmdenseunet":
            model = AcmDenseUnet(self.sess, self.conf, self.is_train)
            self.outputs, self.rates = model.inference(self.inputs)
       
        shape1 = one_hot_annotations.shape
        shape2 = self.outputs.shape
        if shape1[1].value!=shape2[1].value or shape1[2].value!=shape2[2].value:
            self.outputs= tf.image.resize_bilinear(self.outputs, size=(self.output_shape[1],self.output_shape[2]), 
                                                       align_corners=True, name='loss/bilinear')
    
        #——————————————  step：3  ——————————————#
        if self.conf.network_name=="unet" or self.conf.network_name=="denseunet":
            losses = tf.losses.softmax_cross_entropy(one_hot_annotations, self.outputs, scope='loss/losses')
            self.decoded_net_pred = tf.argmax(self.outputs, self.channel_axis, name='accuracy/decode_net_pred')
            self.pred = self.outputs
            
        if self.conf.network_name=="acmdenseunet":
            
            self.net_pred = self.outputs[:,:,:,2:]
            self.decoded_net_pred = tf.argmax(self.net_pred, self.channel_axis, name='accuracy/decode_net_pred')
            losses1 = tf.losses.softmax_cross_entropy(one_hot_annotations, self.net_pred, scope='loss/losses1')
            self.predicted_prob = tf.nn.softmax(self.net_pred, name='softmax')
            
            # CCV 
            self.pred = CCV(self.outputs, self.inputs, 2, 0.5, 1e-8)
            
            lambda1 = 0.01    
            self.pred = tf.squeeze(self.pred)
            losses2 = tf.reduce_sum(tf.square(self.pred-tf.cast(self.annotations,"float32")))
            losses = lambda1*losses1+losses2
            
        #-----------------------------------------------------------------------------------------------------------------#
        #——————————————  step：4  ——————————————#
        self.loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.learning_rate, 
                beta1=self.conf.beta1, beta2=self.conf.beta2, epsilon=self.conf.epsilon)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        
        #——————————————  step：5  ——————————————#
        self.predictions = self.pred
        #self.decoded_predictions = tf.argmax(self.predictions, self.channel_axis, name='accuracy/decode_pred')
        
        gamma = 0.5
        high0 = tf.ones(self.annotations.shape,"int64")
        low0 = tf.zeros(self.annotations.shape,"int64")
        gamma0 = tf.ones(self.annotations.shape)*gamma
        self.decoded_predictions = tf.where(tf.greater_equal(self.predictions,gamma0), high0, low0)
            
        correct_prediction = tf.equal(self.annotations, self.decoded_predictions, name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')
        
        weights = tf.cast(tf.greater(self.decoded_predictions, 0, name='m_iou/greater'),
            tf.int32, name='m_iou/weights')
        self.m_iou, self.miou_op = tf.metrics.mean_iou(self.annotations, self.decoded_predictions, self.conf.class_num,
            weights, name='m_iou/m_ious')
       
        self.out = tf.cast(self.decoded_predictions, tf.float32)
        self.gt = tf.cast(self.annotations, tf.float32)
        
        #——————————————  step：6  ——————————————#
       
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        
        #——————————————  step：7  ——————————————#
       
        trainable_vars = tf.trainable_variables()  
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'batch_norm/moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'batch_norm/moving_variance' in g.name]
        trainable_vars += bn_moving_vars
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        
#———————————————————————————— train —————————————————————————# 
    def train(self):
       
       
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')
        
        if self.conf.reload_epoch > 0:
            self.reload(self.conf.reload_epoch)
     
        train_reader = H5DataLoader(self.conf.data_dir+self.conf.train_data)
        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data)
        
        valid_loss_list = []
        train_loss_list = []
        
        train_acc_list = []
        valid_acc_list = []
       
        train_miou_list = []
        valid_miou_list = []
        
        self.sess.run(tf.local_variables_initializer())
       
        for epoch_num in range(self.conf.max_epoch):
            
           
            if epoch_num % self.conf.test_step == 1:
                inputs, annotations = valid_reader.next_batch(self.conf.batchsize)
                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
                #loss, summary = self.sess.run([self.loss_op, self.valid_summary], feed_dict=feed_dict)
                loss, accuracy, m_iou, _ = self.sess.run([self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
                #self.save_summary(summary, epoch_num)
               
                print(epoch_num, '----valid loss', loss)
                
                # loss
                valid_loss_list.append(loss)
                np.save(self.conf.record_dir+"valid_loss.npy",np.array(valid_loss_list))
                # acc
                valid_acc_list.append(accuracy)
                np.save(self.conf.record_dir+"valid_acc.npy",np.array(valid_acc_list))
                # miou
                valid_miou_list.append(m_iou)
                np.save(self.conf.record_dir+"valid_miou.npy",np.array(valid_miou_list))
                
                #########################################################################
                inputs, annotations = train_reader.next_batch(self.conf.batchsize)
                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: True}
                haha, loss, accuracy, m_iou, _ = self.sess.run([self.train_op, self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
                
                print(epoch_num, '----train loss', loss)
                
                # loss
                train_loss_list.append(loss)
                np.save(self.conf.record_dir+"train_loss.npy",np.array(train_loss_list))
                # acc
                train_acc_list.append(accuracy)
                np.save(self.conf.record_dir+"train_acc.npy",np.array(train_acc_list))
                # miou
                train_miou_list.append(m_iou)
                np.save(self.conf.record_dir+"train_miou.npy",np.array(train_miou_list))
                
            elif epoch_num % self.conf.summary_step == 1:
                inputs, annotations = train_reader.next_batch(self.conf.batchsize)
                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
                #loss, _, summary = self.sess.run([self.loss_op, self.train_op, self.train_summary], feed_dict=feed_dict)
                #self.save_summary(summary, epoch_num)
                #print(epoch_num)
                
               
                #train_loss_list.append(loss)
                #np.save(self.conf.record_dir+"train_loss.npy",np.array(train_loss_list))
            else:
                
                inputs, annotations = train_reader.next_batch(self.conf.batchsize)
                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: True}
                loss,_ = self.sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
                
                print(epoch_num)
            
           
            if epoch_num % self.conf.save_step == 1:
                self.save(epoch_num)
#———————————————————————————— test —————————————————————————#   
    
    def test(self,model_i):
         
        print('---->testing ', model_i)
        
        if model_i > 0:
            self.reload(model_i)
        else:
            print("please set a reasonable test_epoch")
            return
        
        
        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data,False)
        self.sess.run(tf.local_variables_initializer())
       
        losses = []
        accuracies = []
        m_ious = []
        dices = []
        count = 0
        while True:
            inputs, annotations = valid_reader.next_batch(self.conf.batchsize)
           
            if inputs.shape[0] < self.conf.batch:
                break
                
            feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
            loss, accuracy, m_iou, _ = self.sess.run([self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
            print(count)
            print('values----->', loss, accuracy, m_iou)          
            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)
           
            out, gt = self.sess.run([self.out, self.gt], feed_dict=feed_dict)
            
            if self.conf.class_num==2:
                tp = np.sum(out*gt)
                fenmu = np.sum(out)+np.sum(gt)+0.000001
                dice = 2*tp/fenmu
                dices.append(dice)
              
            print('dice----->', dice)
            count+=1
            if count==self.conf.valid_num:
                break
            
        return np.mean(losses),np.mean(accuracies),m_ious[-1],np.mean(dices)
#———————————————————————————— predict —————————————————————————# 
   
    def predict(self):
         
        print('---->predicting ', self.conf.test_epoch)
        
        if self.conf.test_epoch > 0:
            self.reload(self.conf.test_epoch)
        else:
            print("please set a reasonable test_epoch")
            return
        
        test_reader = H5DataLoader(self.conf.data_dir+self.conf.test_data, False)
        self.sess.run(tf.local_variables_initializer())
        predictions = []
        net_predictions = []
        outputs = []
        probabilitys = []
        losses = []
        accuracies = []
        m_ious = []
        
        rate_list = []
        befores = []
        afters = []
        maps = []
        start_maps = []
        count = 0
     
        while True:
            inputs, annotations = test_reader.next_batch(self.conf.batchsize)
           
            if inputs.shape[0] < self.conf.batch:
                break
                
            feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
            loss, accuracy, m_iou, _= self.sess.run([self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
            print('values----->', loss, accuracy, m_iou)
           
            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)
           
            predictions.append(self.sess.run(self.decoded_predictions, feed_dict=feed_dict))
            net_predictions.append(self.sess.run(self.decoded_net_pred, feed_dict=feed_dict))
            outputs.append(self.sess.run(self.outputs, feed_dict=feed_dict))
          
            count+=1
            if count==self.conf.test_num:
                break
       
        print('----->saving outputs')
        print(np.shape(probabilitys))
        np.save(self.conf.sample_dir+"outputs"+".npy",np.array(outputs))
                     
        print('----->saving predictions')
        print(np.shape(predictions))
        num=0
        for index, prediction in enumerate(predictions):
           
            for i in range(prediction.shape[0]):
                np.save(self.conf.sample_dir+"pred"+str(num)+".npy",prediction[i])
                num += 1
                imsave(prediction[i], self.conf.sample_dir + str(index*prediction.shape[0]+i)+'.png')
                
        print('----->saving net_predictions')
        print(np.shape(net_predictions))
        num=0
        for index, prediction in enumerate(net_predictions):
           
            for i in range(prediction.shape[0]):
                np.save(self.conf.sample_dir+"netpred"+str(num)+".npy",prediction[i])
                num += 1
                imsave(prediction[i], self.conf.sample_dir + str(index*prediction.shape[0]+i)+'net.png')
       
        return np.mean(losses),np.mean(accuracies),m_ious[-1]
#———————————————————————————— config_summary —————————————————————————#     
   
    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summarys.append(tf.summary.image(name+'/input', self.inputs, max_outputs=100))
        summarys.append(tf.summary.image(name + '/annotation', tf.cast(tf.expand_dims(
                self.annotations, -1), tf.float32), max_outputs=100))
        summarys.append(tf.summary.image(name + '/prediction', tf.cast(tf.expand_dims(
                self.decoded_predictions, -1), tf.float32), max_outputs=100))
        summary = tf.summary.merge(summarys)
        return summary
#———————————————————————————— save_summary —————————————————————————# 
   
    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)
#———————————————————————————— save —————————————————————————# 
   
    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)
#———————————————————————————— reload —————————————————————————# 
   
    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)
