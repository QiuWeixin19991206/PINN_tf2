#main program
#import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import Model
import numpy as np
import scipy.io
from pyDOE import lhs
import time
import pinn_class_model as PINN
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec  # 散点图
from scipy.interpolate import griddata
from datetime import datetime

#设置空间上限和下限
lb = np.array([-40.0, 0.0])  # 左边界
ub = np.array([39.6875, 10.0])  # 右边界
#load data from NLS.mat
data = scipy.io.loadmat(r'C:\Users\hcxy\Desktop\程序\PT对称耦合非局域\114汇总\非简并单孤子d\自适应\10d.mat')

#切片和分配数据：
t = data['tt'].flatten()[:,None]
x = data['z'].flatten()[:,None]
print(x.shape, t.shape)
Exact1 = data['uu1']
Exact2 = data['uu2']
Exact_u1 = np.real(Exact1)
Exact_v1 = np.imag(Exact1)
Exact_h1 = np.sqrt(Exact_u1**2 + Exact_v1**2)

Exact_u2= np.real(Exact2)
Exact_v2 = np.imag(Exact2)
Exact_h2 = np.sqrt(Exact_u2**2 + Exact_v2**2)
#设置初始、边界和配置点数
N0 = 100
N_b = 100
N_f = 10000

#创建随机索引：
idx_x = np.random.choice(x.shape[0], N0, replace=False)
idx_t = np.random.choice(t.shape[0], N_b, replace=False)

#创建初始数据：
x0 = x[idx_x, :]
t0 = x0*0.0 - 0
u1 = Exact_u1[idx_x, 0:1]
v1 = Exact_v1[idx_x, 0:1]

u2 = Exact_u2[idx_x, 0:1]
v2 = Exact_v2[idx_x, 0:1]

tb = t[idx_t,:]
u1_lb = Exact_u1[0:1, idx_t]  # 因为t=t0，所以去取第一列
u1_ub = Exact_u1[255:256, idx_t]
u2_lb = Exact_u2[0:1, idx_t]  # 因为t=t0，所以去取第一列
u2_ub = Exact_u2[255:256, idx_t]
v1_lb = Exact_v1[0:1, idx_t]  # 因为t=t0，所以去取第一列
v1_ub = Exact_v1[255:256, idx_t]
v2_lb = Exact_v2[0:1, idx_t]  # 因为t=t0，所以去取第一列
v2_ub = Exact_v2[255:256, idx_t]
'''自适应先验点 空变量'''
N_P = 5000
idx_x_p = np.random.choice(x.shape[0], N_P, replace=True)
idx_t_p = np.random.choice(t.shape[0], N_P, replace=True)
x_p = x[idx_x_p, :]
t_p = t[idx_t_p, :]
x_t_p = np.hstack([x_p, t_p])
u1_p = []
v1_p = []
u2_p = []
v2_p = []
for It in range(N_P):
    a = idx_x_p[It]
    b = idx_t_p[It]
    u1_p.append(Exact_u1[a, b])
    v1_p.append(Exact_v1[a, b])
    u2_p.append(Exact_u2[a, b])
    v2_p.append(Exact_v2[a, b])
x_t_p = tf.convert_to_tensor(x_t_p, dtype=tf.float32)
u1_p_tf = tf.convert_to_tensor(u1_p, dtype=tf.float32)
v1_p_tf = tf.convert_to_tensor(v1_p, dtype=tf.float32)
u2_p_tf = tf.convert_to_tensor(u2_p, dtype=tf.float32)
v2_p_tf = tf.convert_to_tensor(v2_p, dtype=tf.float32)
# 增加维度
u1_p_tf = tf.expand_dims(u1_p_tf, axis=1)
v1_p_tf = tf.expand_dims(v1_p_tf, axis=1)
u2_p_tf = tf.expand_dims(u2_p_tf, axis=1)
v2_p_tf = tf.expand_dims(v2_p_tf, axis=1)
#创建共置点：
X_f = lb + (ub - lb) * lhs(2, N_f)#中间点的坐标（x,t）

#将数据转为张量：
x0_t0 = tf.convert_to_tensor(np.concatenate((x0, t0), 1), dtype=tf.float32)#初始点坐标
xlb_tlb = tf.convert_to_tensor(np.concatenate((0.0*tb + lb[0], tb), 1), dtype=tf.float32)
xub_tlb = tf.convert_to_tensor(np.concatenate((0.0*tb + ub[0], tb), 1), dtype=tf.float32)

xf_tf = tf.convert_to_tensor(X_f, dtype=tf.float32)

u1_tf = tf.convert_to_tensor(u1, dtype=tf.float32)
v1_tf = tf.convert_to_tensor(v1, dtype=tf.float32)
u2_tf = tf.convert_to_tensor(u2, dtype=tf.float32)
v2_tf = tf.convert_to_tensor(v2, dtype=tf.float32)
u1_lb_tf = tf.convert_to_tensor(u1_lb, dtype=tf.float32)
u1_ub_tf = tf.convert_to_tensor(u1_ub, dtype=tf.float32)
v1_lb_tf = tf.convert_to_tensor(v1_lb, dtype=tf.float32)
v1_ub_tf = tf.convert_to_tensor(v1_ub, dtype=tf.float32)
u2_lb_tf = tf.convert_to_tensor(u2_lb, dtype=tf.float32)
u2_ub_tf = tf.convert_to_tensor(u2_ub, dtype=tf.float32)
v2_lb_tf = tf.convert_to_tensor(v2_lb, dtype=tf.float32)
v2_ub_tf = tf.convert_to_tensor(v2_ub, dtype=tf.float32)

#创建测试数据：
X, T = np.meshgrid(x, t, sparse=False)
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u1_star = Exact_u1.T.flatten()[:,None]
v1_star = Exact_v1.T.flatten()[:,None]
h1_star = Exact_h1.T.flatten()[:,None]
u2_star = Exact_u2.T.flatten()[:,None]
v2_star = Exact_v2.T.flatten()[:,None]
h2_star = Exact_h2.T.flatten()[:,None]

#创建模型实例：
model = PINN.PINN(lb = lb, ub = ub)

#convert to tensor
X_star_tf = tf.convert_to_tensor(X_star, dtype=tf.float32)

#训练模型：
optimizer = tf.keras.optimizers.Adam()
NUMBER_EPOCHS = 30000# 训练次数

first_start_time = time.time()#当前的时间戳,算总时间
start_time = first_start_time#当前的时间戳，算每次时间
History = []#存loss在列表History中用于画图
loss_point = 0#存loss的记录

# 创建一个空数组
result_list = []
for epoch in range(NUMBER_EPOCHS):
    # print(loss_point)
    with tf.GradientTape() as tape:
        current_loss, f_u1_pred, f_v1_pred, f_u2_pred, f_v2_pred, loss_point= model.loss_fn(x0_t0, xlb_tlb, xub_tlb, xf_tf, u1_tf, v1_tf, u2_tf, v2_tf,
                                     u1_lb_tf, u1_ub_tf, v1_lb_tf, v1_ub_tf, u2_lb_tf, u2_ub_tf, v2_lb_tf, v2_ub_tf, loss_point,epoch,
                                                                                x_t_p, u1_p_tf, v1_p_tf, u2_p_tf, v2_p_tf)

    dW = tape.gradient(current_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(dW, model.trainable_variables))
    if  epoch == 3000 :#epoch % 10 == 0 and
        '''自适应先验点'''
        predictions1 = model(X_star_tf)
        u1_pred = predictions1[:, 0:1]
        v1_pred = predictions1[:, 1:2]
        u1_pred = u1_pred.numpy()
        v1_pred = v1_pred.numpy()
        u2_pred = predictions1[:, 2:3]
        v2_pred = predictions1[:, 3:4]
        u2_pred = u2_pred.numpy()
        v2_pred = v2_pred.numpy()

        # calculate errors:
        error_u1 = (u1_star - u1_pred)**2# / (u1_star)
        error_v1 = (v1_star - v1_pred)**2# / (v1_star)
        error_u2 = (u2_star - u2_pred)**2# / (u2_star)
        error_v2 = (v2_star - v2_pred)**2#/ (v2_star)

        U1 = tf.reshape(error_u1, [51456, 1])
        V1 = tf.reshape(error_v1, [51456, 1])
        U2 = tf.reshape(error_u2, [51456, 1])
        V2 = tf.reshape(error_v2, [51456, 1])
        uuvv = U1 + V1 + U2 + V2
        #值和位置51456
        N_P = 5000
        selected_values1, top_indices1 = tf.math.top_k(uuvv[:, 0], k = N_P)
        top_indices1 = np.array(top_indices1)
        idx_x_p = top_indices1 // 256
        idx_t_p = top_indices1 % 256
        x_p = x[idx_x_p, :]
        t_p = t[idx_t_p, :]
        x_t_p = np.hstack([x_p, t_p])
        u1_p = []
        v1_p = []
        u2_p = []
        v2_p = []
        for It in range(N_P):
            a = idx_x_p[It]
            b = idx_t_p[It]
            u1_p.append(Exact_u1[a, b])
            v1_p.append(Exact_v1[a, b])
            u2_p.append(Exact_u2[a, b])
            v2_p.append(Exact_v2[a, b])
        x_t_p = tf.convert_to_tensor(x_t_p, dtype=tf.float32)
        u1_p_tf = tf.convert_to_tensor(u1_p, dtype=tf.float32)
        v1_p_tf = tf.convert_to_tensor(v1_p, dtype=tf.float32)
        u2_p_tf = tf.convert_to_tensor(u2_p, dtype=tf.float32)
        v2_p_tf = tf.convert_to_tensor(v2_p, dtype=tf.float32)
        # 增加维度
        u1_p_tf = tf.expand_dims(u1_p_tf, axis=1)
        v1_p_tf = tf.expand_dims(v1_p_tf, axis=1)
        u2_p_tf = tf.expand_dims(u2_p_tf, axis=1)
        v2_p_tf = tf.expand_dims(v2_p_tf, axis=1)

        # print(loss_point)
    if epoch % 10 == 0:
        ########存loss在列表History中用于画图
        aaaa = current_loss.numpy()#current_loss由张量转换为np方便存在数组里
        History.append(aaaa)#
        ######每次用时
        prev_time = start_time
        now = time.time()
        edur = datetime.fromtimestamp(now - prev_time) \
                   .strftime("%S.%f")[:-5]
        prev_time = now
        #####总时间
        Total = datetime.fromtimestamp(time.time() - first_start_time) \
            .strftime("%M:%S")
        #####打印
        print(f"epoch = {epoch} "+#训练次数
                         f"elapsed = {Total} " +#总时间
                         f"(+{edur}) " +#每次时间
                         f"loss = {current_loss:.4e} ")#损失函数loss
        start_time = time.time()#刷新时间用于下一轮计算每次用时
    if epoch % 1000 == 0:
        predictions1 = model(X_star_tf)
        u1_pred = predictions1[:, 0:1]
        v1_pred = predictions1[:, 1:2]
        h1_pred = tf.sqrt(u1_pred ** 2 + v1_pred ** 2)
        u1_pred = u1_pred.numpy()
        v1_pred = v1_pred.numpy()
        h1_pred = h1_pred.numpy()

        u2_pred = predictions1[:, 2:3]
        v2_pred = predictions1[:, 3:4]
        h2_pred = tf.sqrt(u2_pred ** 2 + v2_pred ** 2)
        u2_pred = u2_pred.numpy()
        v2_pred = v2_pred.numpy()
        h2_pred = h2_pred.numpy()
        U1_pred = griddata(X_star, u1_pred.flatten(), (X, T), method='cubic')
        V1_pred = griddata(X_star, v1_pred.flatten(), (X, T), method='cubic')
        H1_pred = griddata(X_star, h1_pred.flatten(), (X, T), method='cubic')

        U2_pred = griddata(X_star, u2_pred.flatten(), (X, T), method='cubic')
        V2_pred = griddata(X_star, v2_pred.flatten(), (X, T), method='cubic')
        H2_pred = griddata(X_star, h2_pred.flatten(), (X, T), method='cubic')

        error_u1 = np.linalg.norm(u1_star - u1_pred, 2) / np.linalg.norm(u1_star, 2)
        error_v1 = np.linalg.norm(v1_star - v1_pred, 2) / np.linalg.norm(v1_star, 2)
        error_h1 = np.linalg.norm(h1_star - h1_pred, 2) / np.linalg.norm(h1_star, 2)

        error_u2 = np.linalg.norm(u2_star - u2_pred, 2) / np.linalg.norm(u2_star, 2)
        error_v2 = np.linalg.norm(v2_star - v2_pred, 2) / np.linalg.norm(v2_star, 2)
        error_h2 = np.linalg.norm(h2_star - h2_pred, 2) / np.linalg.norm(h2_star, 2)

        print("u1 error: ", error_u1)
        print("v1 error: ", error_v1)
        print("h1 error: ", error_h1)

        print("u2 error: ", error_u2)
        print("v2 error: ", error_v2)
        print("h2 error: ", error_h2)

        fig = plt.figure(dpi=130)
        ax = plt.subplot(3, 1, 1)
        plt.plot(x, Exact_h1[:, 50], 'b-', linewidth=2, label='Exact')
        plt.plot(x, H1_pred[50, :], 'r--', linewidth=2, label='Prediction')
        ax.set_ylabel('$|U1(t,x)|$')
        ax.set_xlabel('$x$')
        ax.set_title('$t = %.2f$' % (t[50]), fontsize=10)
        plt.legend()

        ax1 = plt.subplot(3, 1, 2)
        plt.plot(x, Exact_h1[:, 100], 'b-', linewidth=2, label='Exact')
        plt.plot(x, H1_pred[100, :], 'r--', linewidth=2, label='Prediction')
        ax1.set_ylabel('$|U1(t,x)|$')
        ax1.set_xlabel('$x$')
        ax1.set_title('$t = %.2f$' % (t[100]), fontsize=10)
        plt.legend()

        ax2 = plt.subplot(3, 1, 3)
        plt.plot(x, Exact_h1[:, 150], 'b-', linewidth=2, label='Exact')
        plt.plot(x, H1_pred[150, :], 'r--', linewidth=2, label='Prediction')
        ax2.set_ylabel('$|U1(t,x)|$')
        ax2.set_xlabel('$x$')
        ax2.set_title('$t = %.2f$' % (t[150]), fontsize=10)
        plt.legend()
        fig.tight_layout()
        plt.savefig('./photo/Q1(t,x)_{}.png'.format(epoch / 1000))

        fig = plt.figure(dpi=130)
        ax = plt.subplot(3, 1, 1)
        plt.plot(x, Exact_h2[:, 50], 'b-', linewidth=2, label='Exact')
        plt.plot(x, H2_pred[50, :], 'r--', linewidth=2, label='Prediction')
        ax.set_ylabel('$|U2(t,x)|$')
        ax.set_xlabel('$x$')
        ax.set_title('$t = %.2f$' % (t[50]), fontsize=10)
        plt.legend()

        ax1 = plt.subplot(3, 1, 2)
        plt.plot(x, Exact_h2[:, 100], 'b-', linewidth=2, label='Exact')
        plt.plot(x, H2_pred[100, :], 'r--', linewidth=2, label='Prediction')
        ax1.set_ylabel('$|U2(t,x)|$')
        ax1.set_xlabel('$x$')
        ax1.set_title('$t = %.2f$' % (t[100]), fontsize=10)
        plt.legend()

        ax2 = plt.subplot(3, 1, 3)
        plt.plot(x, Exact_h2[:, 150], 'b-', linewidth=2, label='Exact')
        plt.plot(x, H2_pred[150, :], 'r--', linewidth=2, label='Prediction')
        ax2.set_ylabel('$|U2(t,x)|$')
        ax2.set_xlabel('$x$')
        ax2.set_title('$t = %.2f$' % (t[150]), fontsize=10)
        plt.legend()
        fig.tight_layout()
        plt.savefig('./photo/Q2(t,x)_{}.png'.format(epoch / 1000))

        fig = plt.figure("预测演化图1", dpi=130, facecolor=None, edgecolor=None, frameon=None)
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(X, T, H1_pred, cmap='coolwarm', rstride=1, cstride=1,
                               linewidth=0, antialiased=False)
        # ax.grid(False)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_zlabel('$|U1(t,x)|$')
        plt.savefig('./photo/预测演化图1_{}.png'.format(epoch / 1000));

        fig = plt.figure("预测演化图2", dpi=130, facecolor=None, edgecolor=None, frameon=None)
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(X, T, H2_pred, cmap='coolwarm', rstride=1, cstride=1,
                               linewidth=0, antialiased=False)
        # ax.grid(False)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_zlabel('$|U2(t,x)|$')
        plt.savefig('./photo/预测演化图2_{}.png'.format(epoch / 1000));

        plt.close("all")


#计算输出：
predictions1 = model(X_star_tf)
model.summary()  # 打印层
print(f" Training time: = {Total} ")  ##输出总的训练时间

u1_pred = predictions1[:,0:1]
v1_pred = predictions1[:,1:2]
h1_pred = tf.sqrt(u1_pred**2 + v1_pred**2)
u1_pred = u1_pred.numpy()
v1_pred = v1_pred.numpy()
h1_pred = h1_pred.numpy()

u2_pred = predictions1[:,2:3]
v2_pred = predictions1[:,3:4]
h2_pred = tf.sqrt(u2_pred**2 + v2_pred**2)
u2_pred = u2_pred.numpy()
v2_pred = v2_pred.numpy()
h2_pred = h2_pred.numpy()

#calculate errors:
error_u1 = np.linalg.norm(u1_star - u1_pred,2)/np.linalg.norm(u1_star,2)
error_v1 = np.linalg.norm(v1_star - v1_pred,2)/np.linalg.norm(v1_star,2)
error_h1 = np.linalg.norm(h1_star - h1_pred,2)/np.linalg.norm(h1_star,2)

error_u2 = np.linalg.norm(u2_star - u2_pred,2)/np.linalg.norm(u2_star,2)
error_v2 = np.linalg.norm(v2_star - v2_pred,2)/np.linalg.norm(v2_star,2)
error_h2 = np.linalg.norm(h2_star - h2_pred,2)/np.linalg.norm(h2_star,2)

print("u1 error: ", error_u1)
print("v1 error: ", error_v1)
print("h1 error: ", error_h1)

print("u2 error: ", error_u2)
print("v2 error: ", error_v2)
print("h2 error: ", error_h2)

#plot results for 0.75s and 1s:
# index = 75
# plt.plot(X_star[0:256,0], h_star[index*256:index*256+256], 'ro', alpha=0.2, label='0.75s actual')
# plt.plot(X_star[0:256,0], h_pred[index*256:index*256+256], 'k', label='0.75s pred.')
#
# index = 100
# plt.plot(X_star[0:256,0], h_star[index*256:index*256+256], 'bo', alpha=0.2, label='1s actual')
# plt.plot(X_star[0:256,0], h_pred[index*256:index*256+256], 'k--', label='1s pred.')
#
# plt.legend()
# plt.show()

U1_pred = griddata(X_star, u1_pred.flatten(), (X, T), method='cubic')
V1_pred = griddata(X_star, v1_pred.flatten(), (X, T), method='cubic')
H1_pred = griddata(X_star, h1_pred.flatten(), (X, T), method='cubic')

U2_pred = griddata(X_star, u2_pred.flatten(), (X, T), method='cubic')
V2_pred = griddata(X_star, v2_pred.flatten(), (X, T), method='cubic')
H2_pred = griddata(X_star, h2_pred.flatten(), (X, T), method='cubic')
# FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
# FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')
import pandas as pd
test = pd.DataFrame(columns=['1'], data=History)
# 2.数据保存，index表示是否显示行名，sep数据分开符
test.to_csv('耦合1.csv', index=False, sep=',')

fig = plt.figure("loss", dpi=100, facecolor=None, edgecolor=None, frameon=True)
plt.plot(range(1, NUMBER_EPOCHS + 1, 10), History, 'r-', linewidth=1, label='learning rate=0.0001')  # + 1前面的数就是迭代次数
plt.xlabel('$\#$ iterations')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend(loc='upper right')
plt.savefig('loss.png')

X0 = np.concatenate((x0, 0 * x0 + lb[1]), 1)  # (x0, 0)
X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
X_u_train = np.vstack([X0, X_lb, X_ub])
xxx = X_f[:, 0]  # a为二维列表ls的第1列
ttt = X_f[:, 1]
plt.figure()
plt.scatter(xxx, ttt, c='red', s=1, label='legend')
plt.xticks(range(-5, 5, 1))
plt.yticks(range(0, 2, 1))
plt.xlabel("x", fontdict={'size': 16})
plt.ylabel("t", fontdict={'size': 16})
plt.title("X_f", fontdict={'size': 20})
plt.legend(loc='best')
plt.savefig('X_f.png')

fig = plt.figure(dpi=130)

ax = plt.subplot(3, 1, 1)
plt.plot(x, Exact_h1[:, 25], 'b-', linewidth=2, label='Exact')
plt.plot(x, H1_pred[25, :], 'r--', linewidth=2, label='Prediction')
ax.set_ylabel('$|Q1(t,x)|$')
ax.set_xlabel('$x$')
ax.set_title('$t = %.2f$' % (t[25]), fontsize=10)
plt.legend()

ax1 = plt.subplot(3, 1, 2)
plt.plot(x, Exact_h1[:, 50], 'b-', linewidth=2, label='Exact')
plt.plot(x, H1_pred[50, :], 'r--', linewidth=2, label='Prediction')
ax1.set_ylabel('$|Q1(t,x)|$')
ax1.set_xlabel('$x$')
ax1.set_title('$t = %.2f$' % (t[50]), fontsize=10)
plt.legend()

ax2 = plt.subplot(3, 1, 3)
plt.plot(x, Exact_h1[:, 100], 'b-', linewidth=2, label='Exact')
plt.plot(x, H1_pred[100, :], 'r--', linewidth=2, label='Prediction')
ax2.set_ylabel('$|Q1(t,x)|$')
ax2.set_xlabel('$x$')
ax2.set_title('$t = %.2f$' % (t[100]), fontsize=10)
plt.legend()
fig.tight_layout()
plt.savefig('Q1(t,x).png')

fig = plt.figure(dpi=130)

ax = plt.subplot(3, 1, 1)
plt.plot(x, Exact_h2[:, 25], 'b-', linewidth=2, label='Exact')
plt.plot(x, H2_pred[25, :], 'r--', linewidth=2, label='Prediction')
ax.set_ylabel('$|Q2(t,x)|$')
ax.set_xlabel('$x$')
ax.set_title('$t = %.2f$' % (t[25]), fontsize=10)
plt.legend()

ax1 = plt.subplot(3, 1, 2)
plt.plot(x, Exact_h2[:, 50], 'b-', linewidth=2, label='Exact')
plt.plot(x, H2_pred[50, :], 'r--', linewidth=2, label='Prediction')
ax1.set_ylabel('$|Q2(t,x)|$')
ax1.set_xlabel('$x$')
ax1.set_title('$t = %.2f$' % (t[50]), fontsize=10)
plt.legend()

ax2 = plt.subplot(3, 1, 3)
plt.plot(x, Exact_h2[:, 100], 'b-', linewidth=2, label='Exact')
plt.plot(x, H2_pred[100, :], 'r--', linewidth=2, label='Prediction')
ax2.set_ylabel('$|Q2(t,x)|$')
ax2.set_xlabel('$x$')
ax2.set_title('$t = %.2f$' % (t[100]), fontsize=10)
plt.legend()
fig.tight_layout()
plt.savefig('Q2(t,x).png')


fig = plt.figure('实际h1(t,x)', dpi=130)
ax = fig.add_subplot(1, 1, 1)
ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'rx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
        clip_on=False)
h = ax.imshow(Exact_h1, interpolation='nearest', cmap='YlGnBu',
              extent=[lb[1], ub[1], lb[0], ub[0]],
              origin='lower', aspect='auto')
plt.colorbar(h)
ax.set_ylabel('$x$')
ax.set_xlabel('$t$')
plt.title('Exact Dynamics1')
plt.savefig('实际h1(t,x).png')

fig = plt.figure('实际h2(t,x)', dpi=130)
ax = fig.add_subplot(1, 1, 1)
ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'rx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
        clip_on=False)
h = ax.imshow(Exact_h2, interpolation='nearest', cmap='YlGnBu',
              extent=[lb[1], ub[1], lb[0], ub[0]],
              origin='lower', aspect='auto')
plt.colorbar(h)
ax.set_ylabel('$x$')
ax.set_xlabel('$t$')
plt.title('Exact Dynamics2')
plt.savefig('实际h2(t,x).png')

fig = plt.figure("实际演化图1", dpi=130, facecolor=None, edgecolor=None, frameon=None)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, Exact_h1.T, cmap='coolwarm', rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
# ax.grid(False)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$|Q1(t,x)|$');
plt.savefig('实际演化图1.png')

fig = plt.figure("预测演化图1", dpi=130, facecolor=None, edgecolor=None, frameon=None)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, H1_pred, cmap='coolwarm', rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
# ax.grid(False)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$|Q1(t,x)|$');
plt.savefig('预测演化图1.png')

fig = plt.figure("实际演化图2", dpi=130, facecolor=None, edgecolor=None, frameon=None)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, Exact_h2.T, cmap='coolwarm', rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
# ax.grid(False)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$|Q2(t,x)|$');
plt.savefig('实际演化图2.png')

fig = plt.figure("预测演化图2", dpi=130, facecolor=None, edgecolor=None, frameon=None)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, H2_pred, cmap='coolwarm', rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
# ax.grid(False)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$|Q2(t,x)|$');
plt.savefig('预测演化图2.png')

scipy.io.savemat('datetf1.mat',
                 {'x': x, 't': t,
                  'U1_pred': h1_pred, 'U2_pred': h2_pred,
                  'loss': History, 'Xstar': X_star,
                  'U1_exact': h1_star, 'U2_exact': h2_star,
                  'error_u1': error_u1, 'error_v1': error_v1, 'error_u2': error_u2, 'error_v2': error_v2,
                  'error_h1': error_h1, 'error_h2': error_h2,
                  })

print("over!")
plt.show()


