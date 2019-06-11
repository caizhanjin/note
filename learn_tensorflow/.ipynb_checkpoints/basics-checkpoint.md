# TensorFlow基础

### 命令式编程和声明式编程区别

+ 命令式编程：命令“机器”如何去做事情(how)，这样不管你想要的是什么(what)，它都会按照你的命令实现。

+ 声明式编程：告诉“机器”你想要的是什么(what)，让机器想出如何去做(how)。

项目|命令式编程|声明式编程
--|--|--
如何执行|立即执行|先构建计算图，再输入数值计算
优点|容易理解、灵活、基准控制、易于调试|先拿到计算图，便于优化。可视化计算图，方便存取
缺点|提供整体优化很难，实现统一的辅助函数很难|很多语言特性用不上，不易于调试，监视中间结果不简单

### 打印tensor值
打印tensor值，必须在会话Session中进行，可以使用：
+ print(sess.run(x)) 
+ print(x.eval())
tensor.eval()=tf.get_default_session().run(tensor)它们的区别主要在于：使用sess.run()能在同一步获取多个tensor中的值。

`
with tf.Session() as sess:
    print(sess.run([x，y]))   #一次能打印两个
    print(x.eval())
    print(y.eval()) #一次只能打印一个
`


### 激活函数
目前TensorFlow提供了7种不同的非线性激活函数，tf.nn.relu、tf.sigmoid和tf.tanh是其中比较常用的几个。
激活函数一般不改变数据的shape，只是加入了非线性因素。
![alt_text](./img/activation.png)

```{.python .input}

```
