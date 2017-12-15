
## TensorFlow简单线性回归（预测房价）


```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001  # 学习率
training_epochs = 1000  # 训练次数
display_step = 50  # 每训练几次输出
```


```python
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
%matplotlib inline
# plt.plot(train_X,train_Y,'ro',label = ['Original data','666'])
plt.plot(train_X,train_Y,'ro',label = 'Original data')
```

![样本坐标](/2017-12-15-TensorFlowLinearRegression/output_2_1.png)



```python
%matplotlib inline
test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
plt.plot(test_X,test_Y,'ro',label = 'data')
```

![png](/2017-12-15-TensorFlowLinearRegression/output_3_1.png)



```python
# 训练样本数
n_samples = train_X.shape[0]

# 创建输入占位符
X = tf.placeholder('float')
Y = tf.placeholder('float')

# 需要更新的权重
W = tf.Variable(np.random.randn(),name='Weight')  # 随机数初始化
b = tf.Variable(np.random.randn(), name="bias")
'''
# numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。
# numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中。 
'''
# 线性函数
predict = tf.add(tf.multiply(W,X),b)

# cost为标准差
# cost = tf.pow(tf.reduce_sum(predict-Y)/n_samples,2)
# cost 平方
cost = tf.reduce_sum((predict-Y)**2)/n_samples

# 优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# 初始化全部变量
init = tf.global_variables_initializer()
```


```python
# 执行计算图
with tf.Session() as sess:
    sess.run(init)
    # 计算
    for epoch in range(training_epochs):
        _,c = sess.run([optimizer,cost],feed_dict = {X:train_X,Y:train_Y})
        if (epoch+1) % display_step == 0:
            c = sess.run(cost,feed_dict = {X:train_X,Y:train_Y})
            print('cost:',c,'epoch:',epoch+1)

    # 计算结束，输出结果
    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    
    %matplotlib inline
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    # Testing example, as requested (Issue #2)
    
    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.pow(tf.reduce_sum(predict - Y) / (test_X.shape[0]),2),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))
```

    cost: 0.158196 epoch: 50
    cost: 0.156174 epoch: 100
    cost: 0.156118 epoch: 150
    cost: 0.156064 epoch: 200
    cost: 0.156011 epoch: 250
    cost: 0.15596 epoch: 300
    cost: 0.155909 epoch: 350
    cost: 0.15586 epoch: 400
    cost: 0.155812 epoch: 450
    cost: 0.155765 epoch: 500
    cost: 0.155719 epoch: 550
    cost: 0.155675 epoch: 600
    cost: 0.155631 epoch: 650
    cost: 0.155589 epoch: 700
    cost: 0.155547 epoch: 750
    cost: 0.155507 epoch: 800
    cost: 0.155467 epoch: 850
    cost: 0.155428 epoch: 900
    cost: 0.155391 epoch: 950
    cost: 0.155354 epoch: 1000
    Optimization Finished!
    Training cost= 0.155354 W= 0.267142 b= 0.688864 
    
    Testing... (Mean square loss Comparison)
    Testing cost= 0.00219555
    Absolute mean square loss difference: 0.153158

![png](/2017-12-15-TensorFlowLinearRegression/output_5_1.png)

