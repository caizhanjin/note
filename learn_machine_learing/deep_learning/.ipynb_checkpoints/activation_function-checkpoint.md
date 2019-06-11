# 激活函数

总结：激活函数是用来加入非线性因素的，实现去线性化，提高神经网络对模型的表达能力，解决线性模型所不能解决的问题。

特征：具有单调性，非线性函数。

+ Sigmoid
+ tanh
+ ReLU
+ Leaky ReLU
+ Maxout
+ ELU

## Sigmoid
+ 输入非常大或非常小时没有梯度
+ 输出均值非0
+ Exp计算复杂，速度就会变慢
+ 梯度消失
![alt_text](./img/sigmod.png)

## Tanh
+ 依旧没有梯度
+ 输出均值是0
+ 计算也是复杂
![alt_text](./img/tanh.png)

## ReLU 
+ 不饱和（梯度不会过小）
+ 计算量小
+ 收敛速度快
+ 输出均值非0
+ Dead ReLU：一个非常大的梯度流过神经元，不会再对数据有激活现象了（小于零时，梯度得不到更新）
![alt_text](./img/relu.png)

## Leaky-ReLU
+ 解决ReLU的问题
![alt_text](./img/leaky_relu.png)

## ELU
+ 均值更接近0
+ 小于0时，计算量大
![alt_text](./img/elu.png)

## Maxout
+ ReLU 的泛化版本
+ 没有deal relu
+ 但是参数会翻倍

## 使用激活函数技巧
+ Relu小心设置learning rate
+ 不要使用sigmoid（过大、慢）
+ 使用Leaky ReLU、maxout、ELU
+ 可以试试tanh，但不要抱太大期望（计算量大）

```{.python .input}

```
