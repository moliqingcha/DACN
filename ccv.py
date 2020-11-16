from keras import backend as K
import numpy as np
import tensorflow as tf

def CCV(net_output, inputs0, max_iter, gamma, epsilon): 
    
    #---------------------------------------- convexified CV ------------------------------------------------#
       
    batchsize = net_output.shape[0].value
    height = net_output.shape[1].value
    width = net_output.shape[2].value
    
    mu = tf.expand_dims(net_output[:,:,:,0],-1)*0.001
    lambda0 = tf.expand_dims(net_output[:,:,:,1],-1)*0.001
    inputs0 = tf.expand_dims(inputs0[:,:,:,0], -1)
   
    Gx=tf.constant(np.array([-1,1,0]))
    Gx=tf.cast( Gx,"float32")
    filterx = tf.reshape(Gx, [3,1,1,1])
    Gy=tf.constant(np.array([-1,1,0]))
    Gy=tf.cast( Gy,"float32")
    filtery = tf.reshape(Gy, [1,3,1,1])
    laplace = tf.constant(np.array([[0,1,0],[1,0,1],[0,1,0]]))
    laplace = tf.cast( laplace,"float32")
    filterlaplace = tf.reshape(laplace, [3,3,1,1])
    
    C1=1
    C2=0
    
    net_pred = net_output[:,:,:,2:]      
    predicted_prob = tf.nn.softmax(net_pred)
    u = tf.expand_dims(predicted_prob[:,:,:,1], -1)
    dx=tf.zeros([batchsize, height, width,1])
    dy=tf.zeros([batchsize, height, width,1])
    bx=tf.zeros([batchsize, height, width,1])
    by=tf.zeros([batchsize, height, width,1])
        
    for i in range(max_iter):
        print(i)
        #----------------------update u------------------------#
        r = tf.square(inputs0-C1)-tf.square(inputs0-C2)
        bx_partial = tf.nn.conv2d(bx, filterx, strides=[1,1,1,1], padding='SAME')
        by_partial = tf.nn.conv2d(by, filtery, strides=[1,1,1,1], padding='SAME')
        dx_partial = tf.nn.conv2d(dx, filterx, strides=[1,1,1,1], padding='SAME')
        dy_partial = tf.nn.conv2d(dy, filtery, strides=[1,1,1,1], padding='SAME')
        alpha=bx_partial+by_partial-dx_partial-dy_partial
        temp = tf.multiply(tf.div(mu,lambda0+epsilon),r)
        beta=0.25*(tf.nn.conv2d(u, filterlaplace, strides=[1,1,1,1], padding='SAME')+alpha-temp)
        u=beta
        high0 = tf.ones([batchsize, height, width,1])
        u = tf.where(tf.greater(u,high0), high0, u)
        low0 = tf.zeros([batchsize, height, width,1])
        u = tf.where(tf.greater(low0,u), low0, u)
            
        #----------------------update d--------------------------#
        Ix = tf.nn.conv2d(u, filterx, strides=[1,1,1,1], padding='SAME')
        Iy = tf.nn.conv2d(u, filtery, strides=[1,1,1,1], padding='SAME')
        tempx1=tf.abs(Ix+bx)-tf.div(high0,lambda0+epsilon)
        tempx1 = tf.where(tf.greater(low0,tempx1), low0, tempx1)
        tempx2=tf.sign(Ix+bx)
        dx=tf.multiply(tempx1,tempx2)
            
        tempy1=tf.abs(Iy+by)-tf.div(high0,lambda0+epsilon)
        tempy1 = tf.where(tf.greater(low0,tempy1), low0, tempy1)
        tempy2=tf.sign(Iy+by)
        dy=tf.multiply(tempy1,tempy2)
            
        #----------------------update b-------------------------#
        bx=bx+Ix-dx
        by=by+Iy-dy
            
        #----------------------update C1,C2--------------------------#
        gamma0 = tf.ones([batchsize, height, width,1])*gamma
        region_in = tf.where(tf.greater_equal(u,gamma0), high0, low0)
        C1 = tf.reduce_sum(tf.multiply(region_in,inputs0))/(tf.reduce_sum(region_in)+epsilon)
            
        region_out = tf.where(tf.less(u,gamma0), high0, low0)
        C2 = tf.reduce_sum(tf.multiply(region_out,inputs0))/(tf.reduce_sum(region_out)+epsilon)
     
    pred1 = u      
    
    return pred1
    