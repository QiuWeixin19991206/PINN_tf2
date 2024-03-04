import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import Model,layers
import numpy as np
#create model class
class PINN(Model):
    def __init__(self, lb, ub):
        super(PINN, self).__init__()
        
        #create layers: [2 40 40 40 40 2]
        self.fc1 = Dense(40, input_shape=(None,2), activation = 'tanh', kernel_initializer = 'glorot_uniform')#初始
        self.bn1 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc2 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')#中间隐藏层
        self.bn2 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc3 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.bn3 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc4 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.bn4 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc5 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.bn5 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc6 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.bn6 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)

        self.fc7 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        self.output_layer = Dense(8, activation = 'linear', kernel_initializer = 'glorot_uniform')#输出层




        # self.fcs1 = Dense(40, input_shape=(None,2), activation = 'tanh', kernel_initializer = 'glorot_uniform')#初始
        # self.bn1 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs2 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')#中间隐藏层
        # self.bn2 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs3 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        # self.bn3 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs4 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        # self.bn4 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs5 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        # self.bn5 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs6 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        # self.bn6 = layers.BatchNormalization(axis=-1,center=True,scale=True,trainable=True)
        #
        # self.fcs7 = Dense(40, activation = 'tanh', kernel_initializer = 'glorot_uniform')
        # self.output_layer = Dense(4, activation = 'linear', kernel_initializer = 'glorot_uniform')#输出层

        self.lb = lb
        self.ub = ub
    
    #call method
    def call(self, x):
        return self.net(x)

    #返回 u，v 和导数的类方法
    def net_uv1(self, x):
        with tf.GradientTape(persistent=True) as tape_1:
            tape_1.watch(x)
            net_output= self.net(x)
            u1 = net_output[:,0:1]
            v1 = net_output[:,1:2]
            u2 = net_output[:, 2:3]
            v2 = net_output[:, 3:4]
            u1x = net_output[:,4:5]
            v1x = net_output[:,5:6]
            u2x = net_output[:, 6:7]
            v2x = net_output[:, 7:8]
        del tape_1
        return u1, v1, u2, v2, u1x, v1x, u2x, v2x

    
    #返回偏微分方程残差的类方法
    def net_f_uv(self, x, ):
        _,  _,  _,  _, real1, image1, real2, image2 = self.net_uv1(x)
        with tf.GradientTape(persistent=True) as tape_2:
            tape_2.watch(x)
            with tf.GradientTape(persistent=True) as tape_3:
                tape_3.watch(x)
                output1 = self.net(x)
                u1 = output1[:, 0:1]
                v1 = output1[:, 1:2]
                u2 = output1[:, 2:3]
                v2 = output1[:, 3:4]

            u1_x = tape_3.gradient(u1, x)[:,0:1]
            v1_x = tape_3.gradient(v1, x)[:,0:1]
            u1_t = tape_3.gradient(u1, x)[:,1:2]
            v1_t = tape_3.gradient(v1, x)[:,1:2]

            u2_x = tape_3.gradient(u2, x)[:,0:1]
            v2_x = tape_3.gradient(v2, x)[:,0:1]
            u2_t = tape_3.gradient(u2, x)[:,1:2]
            v2_t = tape_3.gradient(v2, x)[:,1:2]

        u1_xx = tape_2.gradient(u1_x, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:,0:1]
        v1_xx = tape_2.gradient(v1_x, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:,0:1]

        u2_xx = tape_2.gradient(u2_x, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:,0:1]
        v2_xx = tape_2.gradient(v2_x, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)[:,0:1]

        sigma = 1

        # f_u1 = 2 * sigma * image1 * v1 ** 3 - 6 * sigma * real1 * u1 * v1 ** 2 - 6 * sigma * image1 * u1 ** 2 * v1 - 2 * sigma * image2 * u2 ** 2 * v1 - 4 * sigma * real2 * u2 * v2 * v1 + 2 * sigma * image2 * v2 ** 2 * v1 + 2 * sigma * real1 * u1 ** 3 + 2 * sigma * real2 * u2 ** 2 * u1 - 4 * sigma * image2 * u2 * v2 * u1 - 2 * sigma * real2 * v2 ** 2 * u1 - v1_t + u1_xx
        #
        # f_v1 = -2 * sigma * real1 * v1 ** 3 - 6 * sigma * image1 * v1 ** 2 * u1 + 6 * sigma * real1 * u1 ** 2 * v1 + 2 * sigma * real2 * u2 ** 2 * v1 - 4 * sigma * image2 * u2 * v2 * v1 - 2 * sigma * real2 * v2 ** 2 * v1 + 2 * sigma * image1 * u1 ** 3 + 2 * sigma * image2 * u2 ** 2 * u1 + 4 * sigma * real2 * u2 * v2 * u1 - 2 * sigma * image2 * v2 ** 2 * u1 + v1_xx + u1_t
        #
        # f_u2 = -2 * sigma * real1 * v1 ** 2 * u2 + 2 * sigma * image1 * v1 ** 2 * v2 - 4 * sigma * image1 * u1 * v1 * u2 - 4 * sigma * real1 * u1 * v1 * v2 + 2 * sigma * real1 * u1 ** 2 * u2 - 2 * sigma * image1 * u1 ** 2 * v2 + 2 * sigma * real2 * u2 ** 3 - 6 * sigma * image2 * u2 ** 2 * v2 - 6 * sigma * real2 * u2 * v2 ** 2 + 2 * sigma * image2 * v2 ** 3 - v2_t + u2_xx
        #
        # f_v2 = -2 * sigma * image1 * v1 ** 2 * u2 - 2 * sigma * real1 * v1 ** 2 * v2 + 4 * sigma * real1 * u1 * v1 * u2 - 4 * sigma * image1 * u1 * v1 * v2 + 2 * sigma * image1 * u1 ** 2 * u2 + 2 * sigma * real1 * u1 ** 2 * v2 + 2 * sigma * image2 * u2 ** 3 + 6 * sigma * real2 * u2 ** 2 * v2 - 6 * sigma * image2 * v2 ** 2 * u2 - 2 * sigma * real2 * v2 ** 3 + v2_xx + u2_t
        f_u1 = 2 * sigma * image2 * u2 * v1 + 2 * sigma * real2 * u2 * u1 - 2 * sigma * real1 * v1 ** 2 + 4 * sigma * image1 * u1 * v1 - 2 * sigma * real2 * v2 * v1 + 2 * sigma * real1 * u1 ** 2 + 2 * sigma * image2 * v2 * u1 - v1_t + u1_xx
        f_v1 = 2 * sigma * real2 * u2 * v1 - 2 * sigma * image2 * u2 * u1 + 2 * sigma * image1 * v1 ** 2 + 4 * sigma * real1 * u1 * v1 + 2 * sigma * image2 * v2 * v1 - 2 * sigma * image1 * u1 ** 2 + 2 * sigma * real2 * v2 * u1 + v1_xx + u1_t
        f_u2 = 2 * sigma * real2 * u2 ** 2 + 2 * sigma * image1 * v1 * u2 + 2 * sigma * real1 * u1 * u2 + 4 * sigma * image2 * u2 * v2 - 2 * sigma * real1 * v1 * v2 + 2 * sigma * image1 * u1 * v2 - 2 * sigma * real2 * v2 ** 2 - v2_t + u2_xx
        f_v2 = -2 * sigma * image2 * u2 ** 2 + 2 * sigma * real1 * v1 * u2 - 2 * sigma * image1 * u1 * u2 + 4 * sigma * real2 * u2 * v2 + 2 * sigma * image1 * v1 * v2 + 2 * sigma * real1 * u1 * v2 + 2 * sigma * image2 * v2 ** 2 + u2_t + v2_xx

        del tape_2, tape_3
        return f_u1, f_v1, f_u2, f_v2

    #网络前向传播方法
    def net(self, x):
        x1 = 2.0 * (x - self.lb)/(self.ub - self.lb) - 1.0
        x1= self.fc1(x1)
        # x = self.bn1(x)
        x1 = self.fc2(x1)
        # x = self.bn2(x)
        x1 = self.fc3(x1)
        # x = self.bn3(x)
        x1 = self.fc4(x1)
        # x = self.bn4(x)
        x1 = self.fc5(x1)
        # x = self.bn5(x)
        x1 = self.fc6(x1)
        # x = self.bn6(x)

        # x2 = 2.0 * (x - self.lb)/(self.ub - self.lb) - 1.0
        # x2 = self.fcs1(x2)
        # # x = self.bn1(x)
        # x2 = self.fcs2(x2)
        # # x = self.bn2(x)
        # x2 = self.fcs3(x2)
        # # x = self.bn3(x)
        # x2 = self.fcs4(x2)
        # # x = self.bn4(x)
        # x2 = self.fcs5(x2)
        # # x = self.bn5(x)
        # x2 = self.fcs6(x2)
        # # x = self.bn6(x)

        return self.output_layer(self.fc7(x1))

    #loss function method
    def loss_fn(self, x0_t0, xlb_tlb, xub_tub, xf_tf, u1, v1, u2, v2,
                u1_lb_tf, u1_ub_tf, v1_lb_tf, v1_ub_tf, u2_lb_tf, u2_ub_tf, v2_lb_tf, v2_ub_tf, loss_point, epoch,
                x_t_p, u1_p_tf, v1_p_tf, u2_p_tf, v2_p_tf):
        u1_pred, v1_pred, u2_pred, v2_pred, _, _, _, _ = self.net_uv1(x0_t0)
        u1_lb_pred, v1_lb_pred, u2_lb_pred, v2_lb_pred, u2_x_lb_pred, v2_x_lb_pred, u1_x_lb_pred, v1_x_lb_pred = self.net_uv1(xlb_tlb)
        u1_ub_pred, v1_ub_pred, u2_ub_pred, v2_ub_pred, u2_x_ub_pred, v2_x_ub_pred, u1_x_ub_pred, v1_x_ub_pred = self.net_uv1(xub_tub)
        f_u1_pred, f_v1_pred, f_u2_pred, f_v2_pred= self.net_f_uv(xf_tf)

        loss1 = tf.reduce_mean(tf.square(u1_pred - u1)) + \
        tf.reduce_mean(tf.square(v1_pred - v1)) + \
        tf.reduce_mean(tf.square(u2_pred - u2)) + \
        tf.reduce_mean(tf.square(v2_pred - v2))
        loss2 = tf.reduce_mean(tf.square(u2_lb_tf - u2_lb_pred)) + \
        tf.reduce_mean(tf.square(v2_lb_tf - v2_lb_pred)) + \
        tf.reduce_mean(tf.square(u2_ub_tf - u2_ub_pred)) + \
        tf.reduce_mean(tf.square(v2_ub_tf - v2_ub_pred)) + \
        tf.reduce_mean(tf.square(u1_lb_tf - u1_lb_pred)) + \
        tf.reduce_mean(tf.square(v1_lb_tf - v1_lb_pred)) + \
        tf.reduce_mean(tf.square(u1_ub_tf - u1_ub_pred)) + \
        tf.reduce_mean(tf.square(v1_ub_tf - v1_ub_pred))
        loss3 = (tf.reduce_mean(tf.square(f_u1_pred)) + \
        tf.reduce_mean(tf.square(f_v1_pred)) + \
        tf.reduce_mean(tf.square(f_u2_pred)) + \
        tf.reduce_mean(tf.square(f_v2_pred)))


        print(f"IC = {loss1:.4e} " +
              f"BC = {loss2:.4e} " +
              f"PDE = {loss3:.4e} " +
              f"POINT = {loss_point:.4e} "
              )

        if epoch > 3000:
            u1_pred, v1_pred, u2_pred, v2_pred, _, _, _, _ = self.net_uv1(x_t_p)
            loss = tf.reduce_mean(tf.square(u1_pred - u1_p_tf)) + \
                         tf.reduce_mean(tf.square(v1_pred - v1_p_tf)) + \
                         tf.reduce_mean(tf.square(u2_pred - u2_p_tf)) + \
                         tf.reduce_mean(tf.square(v2_pred - v2_p_tf))
        else:
            loss = loss_point + loss1 + loss2 + loss3


        return loss, f_u1_pred, f_v1_pred, f_u2_pred, f_v2_pred, loss_point