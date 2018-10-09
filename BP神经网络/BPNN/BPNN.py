#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/518:04
# @Author  : DaiPuWei
# E-Mail   : 771830171@qq.com
# @Site    : 中国民航大学北教17实验室506
# @File    : BPNN.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

class BPNN(object):
    def __init__(self,input_n,hidden_n,output_n,lambd):
        """
        这是BP神经网络类的构造函数
        :param input_n:输入层神经元个数
        :param hidden_n: 隐藏层神经元个数
        :param output_n: 输出层神经元个数
        :param lambd: 正则化系数
        """
        self.Train_Data = tf.placeholder(tf.float64,shape=(None,input_n),name='input_dataset')                                  # 训练数据集
        self.Train_Label = tf.placeholder(tf.float64,shape=(None,output_n),name='input_labels')                                 # 训练数据集标签
        self.input_n = input_n                                                                                                    # 输入层神经元个数
        self.hidden_n = hidden_n                                                                                                  # 隐含层神经元个数
        self.output_n = output_n                                                                                                  # 输出层神经元个数
        self.lambd = lambd                                                                                                        # 正则化系数
        self.input_weights = tf.Variable(tf.random_normal((self.input_n, self.hidden_n),mean=0,stddev=1,dtype=tf.float64),trainable=True)                                       # 输入层与隐含层之间的权重
        self.hidden_weights =  tf.Variable(tf.random_normal((self.hidden_n,self.output_n),mean=0,stddev=1,dtype=tf.float64),trainable=True)                                      # 隐含层与输出层之间的权重
        self.hidden_threshold = tf.Variable(tf.random_normal((1,self.hidden_n),mean=0,stddev=1,dtype=tf.float64),trainable=True)                                            # 隐含层的阈值
        self.output_threshold = tf.Variable(tf.random_normal((1,self.output_n),mean=0,stddev=1,dtype=tf.float64),trainable=True)                                            # 输出层的阈值
        # 将层与层之间的权重与偏置项加入损失集合
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.lambd)(self.input_weights))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.lambd)(self.hidden_weights))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.lambd)(self.hidden_threshold))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.lambd)(self.output_threshold))
        # 定义前向传播过程
        self.hidden_cells = tf.sigmoid(tf.matmul(self.Train_Data,self.input_weights)+self.hidden_threshold)
        self.output_cells = tf.sigmoid(tf.matmul(self.hidden_cells,self.hidden_weights)+self.output_threshold)
        # 定义损失函数,并加入损失集合
        self.MSE = tf.reduce_mean(tf.square(self.output_cells-self.Train_Label))
        tf.add_to_collection('loss',self.MSE)
        # 定义损失函数,均方误差加入L2正则化
        self.loss = tf.add_n(tf.get_collection('loss'))

    def train_test(self,Train_Data,Train_Label,Test_Data,Test_Label,learn_rate,epoch,iteration,batch_size):
        """
        这是BP神经网络的训练函数
        :param Train_Data: 训练数据集
        :param Train_Label: 训练数据集标签
        :param Test_Data: 测试数据集
        :param Test_Label: 测试数据集标签
        :param learn_rate:  学习率
        :param epoch:  时期数
        :param iteration: 一个epoch的迭代次数
        :param batch_size:  小批量样本规模
        """
        train_loss = []                 # 训练损失
        test_loss = []                  # 测试损失
        test_accarucy = []              # 测试精度
        with tf.Session() as sess:
            datasize = len(Train_Label)
            self.train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
            sess.run(tf.global_variables_initializer())
            for e in np.arange(epoch):
                for i in range(iteration):
                    start = (i*batch_size)%datasize
                    end = np.min([start+batch_size,datasize])
                    sess.run(self.train_step,
                             feed_dict={self.Train_Data:Train_Data[start:end],self.Train_Label:Train_Label[start:end]})
                    if i % 10000 == 0:
                        total_MSE = sess.run(self.MSE,
                                             feed_dict={self.Train_Data:Train_Data,self.Train_Label:Train_Label})
                        print("第%d个epoch中，%d次迭代后，训练MSE为:%g"%(e+1,i+10000,total_MSE))
                # 训练损失
                _train_loss = sess.run(self.MSE,feed_dict={self.Train_Data:Train_Data,self.Train_Label:Train_Label})
                train_loss.append(_train_loss)
                # 测试损失
                _test_loss = sess.run(self.MSE, feed_dict={self.Train_Data:Test_Data, self.Train_Label: Test_Label})
                test_loss.append(_test_loss)
                # 测试精度
                test_result = sess.run(self.output_cells,feed_dict={self.Train_Data:Test_Data})
                test_accarucy.append(self.Accuracy(test_result,Test_Label))
        return train_loss,test_loss,test_accarucy

    def Accuracy(self,test_result,test_label):
        """
        这是BP神经网络的测试函数
        :param test_result: 测试集预测结果
        :param test_label: 测试集真实标签
        """
        predict_ans = []
        label = []
        for (test,_label) in zip(test_result,test_label):
            test = np.exp(test)
            test = test/np.sum(test)
            predict_ans.append(np.argmax(test))
            label.append(np.argmax(_label))
        return accuracy_score(label,predict_ans)