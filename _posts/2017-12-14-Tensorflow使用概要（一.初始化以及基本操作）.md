# 什么是TF

一个使用数据流图的开源**数值计算**软件库，Google研发的一个深度神经网路框架

__工具包__: TF Learn, TF Slim, 更高级的API封装——`Keras`、`Pretty`、`Tensor`

__数据流图__: 包括节点、线，就像流程图一样，数据在图中flow，我们就可以控制数据在图中的每个节点具体进行哪种计算，同时也能对计算过程进行优化，例如并行计算的各个节点可以分配多进程（或者分布式）进行并行计算。张量从图中流过的直观图像——`tensorflow。`

```python
import tensorflow as tf
a = tf.add(2,3)  # 如果没有给定变量名，tf会随机赋值变量名，这里2,3也是有变量名的
print(a)  # tf 的tensor对象
sess = tf.Session()
print(sess.run(a))  # tf.add(2,3) 的计算结果
sess.close()

with tf.Session() as sess:
    print(sess.run(a))
    sess.run([a,b,c,d]) # 可以传多个值
'''这里的代码都默认在一个图中'''
```

###构建多个图

**不推荐** ：需要多个session，都占用默认资源，只能通过`python`、`numpy`的方式进行数据交互，丧失了分布式的部分好处

```python
g = tf.Graph() # 创建一个图
'''在图中运行代码'''
with g.as_default():
    x = tf.add(2,3)

sess = tf.Session(graph = g)
with tf.Session() as sess:
    sess.run(x)
    
'''获取默认的图'''
g = tf.get_default_graph()

'''查看图的定义'''
my_const = tf.constant([1.0,2.0],name = 'my_const')  # 固定值
with tf.Session() as sess:
    print(sess.graph.as_graph_def())   
    # 结果就是这个图的定义，类似于一个多级的json字符串
```



# Tensor 类型

数据类型：和python类似，略

常量：constant，占据的空间在定义的时候产生，常量内容特别大的时候，很浪费资源，如果不是特别需要，尽量用变量替代。

变量：Variable、get_variable

# Tensorflow基本操作

####Tensor board

可以用来查看图是否符合预期

```python
import tensorflow as tf
a = tf.constant(2,name='a')  # 定义常量,这里如果不定义name，图的定义中的变量名是随机的
b = tf.constant(2,name='b')
res_sum = tf.add(a,b) # 计算
with tf.Session() as sess:
    # 将一行添加到 Tensor board
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    print(sess.run(res_sum))    # 执行计算
    writer.close()  # 关闭 writer
```

执行py文件，然后命令行启动 tensorboard

```
python tfbd.py
tensorboard --logdir ='tfbd.py' --port 8008
```

然后浏览器打开这个端口查看结果。

*道理是这么个道理，但是由于用的是小破Windows本，出现警告：`Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2` 并没有执行成功，具体原因也懒得解决了，等换到 linux下再具体解决*

--------------



#### 创建常量：

```python
tf.constant(value,dtype=None,shape=None,name='Const',verify_shape=False)
```

*********



#### 创建变量：

```python
tf.Variable(self,initial_value=None,trainable=True,collections=None,
               validate_shape=True,caching_device=None,name=None,
               variable_def=None,dtype=None,expected_shape=None,
               import_scope=None,constraint=None)
tf.get_variable(name,shape=None,dtype=None,initializer=None,regularizer=None,
                 trainable=True,collections=None,caching_device=None,
                 partitioner=None,validate_shape=True,use_resource=None,
                 custom_getter=None,constraint=None)
# e.g.  # 另：这里的 value 都可以用下面的创建初始值的方式赋值
W1 = tf.get_variable(     # 与下一个等同
    "W1", [25,12288],
    initializer = 
    tf.contrib.layers.xavier_initializer(seed = 1))
tf.Variable(                      # 与上一个等同
  	initial_value=
    tf.contrib.layers.xavier_initializer(seed = 1)(shape=(25, 12288)),
    dtype=tf.float32,name="W1")

# 另：变量必须有初始化操作
# 初始化全局变量：
init = tf.global_variables_initializer() # Initialize all the variables
with tf.Session() as sess: # Start the session to compute the graph
    sess.run(init) # Run the initialization
    ...
# 初始化单个变量：
W1=tf.Variable(tf.truncated_normal([700,10]))
with tf.Session() as sess:
    sess.run(W1.initializer)
    print(W1.eval())
    W1.assign(123)
    print(W1.eval())
    ...
    
'''变量赋值'''
# assign
W1=tf.Variable(10)
with tf.Session() as sess:
    sess.run(W1.initializer)
    print(W1.eval())            # 10
    sess.run(W1.assign(1))
    print(W1.eval())            # 1
    sess.run(W1.assign_add(1))
    print(W1.eval())            # 2
    sess.run(W1.assign_sub(1))
    print(W1.eval())            # 1
    WW = W1.assign(123)  
    sess.run(WW)
    print(W1.eval())            # 123
```

-----------------



#### 指定初始值：

```python
tf.zeros(shape, dtype=dtypes.float32, name=None)
	# tf.zeros([2,3])
  
tf.zeros_like(tensor, dtype=None, name=None, optimize=True)
	# tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
	# tf.zeros_like(tensor)  # [[0, 0, 0], [0, 0, 0]]
	# optimize: if true, attempt to statically determine the shape of 'tensor' and encode it as a constant.
    
tf.fill(dims, value, name=None)
	# fill([2, 3], 9) ==> [[9, 9, 9],[9, 9, 9]]
  
tf.linspace(start, stop, num, name=None)
	# tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
  
tf.range(start, limit=None, delta=1, dtype=None, name="range")
	# tf.range(3, 18, 3)  # [3, 6, 9, 12, 15]
	# tf.range(3, 1, -0.5)  # [3, 2.5, 2, 1.5]
	# tf.range(5)  # [0, 1, 2, 3, 4]

'''随机数'''
tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=dtypes.float32,
                 seed=None,name=None)
tf.truncated_normal(shape,mean=0.0,stddev=1.0,dtype=dtypes.float32,
                    seed=None,name=None)
tf.random_uniform(shape,minval=0,maxval=None,dtype=dtypes.float32,
                   seed=None,name=None)
tf.random_shuffle(value, seed=None, name=None)
#   [[1, 2],[3, 4],[5, 6]]  ==> [[5, 6],[1, 2],[3, 4]]
tf.random_crop(value, size, seed=None, name=None)
# Returns:A cropped tensor of the same rank as `value` and shape `size`.
tf.multinomial(logits, num_samples, seed=None, name=None)
tf.random_gamma(shape,alpha,beta=None,dtype=dtypes.float32,
                 seed=None,name=None):
```

------------------------



#### 多个 session 之间数据不共享

多个session可以对同一个变量分别初始化、赋值等等，不会相互影响。

#### 变量依赖

如果一个变量需要使用另一个变量，但是这个变量没有初始化，那么就会报错。

```python
# 解决方法
w = tf.Variable(tf.truncated_normal([700,10]))
u = tf.Variable(2*w.intialized_value()) # 显式的初始化一下依赖的变量
```

#### 交互式session

```python
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(4.0)
c = a*b
print(c.eval())
sess.close()
'''其实和普通session没有太大的差别'''
```

#### 依赖

```python
# tf.Graph.control_dependencies(control_input)
with g.control_dependencies([a, b, c]):
  # `d` and `e` will only run after `a`, `b`, and `c` have executed.
  d = ...
  e = ...
```

