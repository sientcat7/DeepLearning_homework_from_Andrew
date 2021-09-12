#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('C:\\Users\\SilentCat\\DeepLearning-Andrew\\第二章\\datasets\\train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('C:\\Users\\SilentCat\DeepLearning-Andrew\\第二章\\datasets\\test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    # classes保存的是以bytes类型保存的两个字符串数据，分别是：[b'non-cat', b'cat']
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# In[2]:


x_train, y_train, x_test, y_test, classes = load_dataset()


# In[3]:


X_train = np.array(x_train)
Y_train = np.array(y_train)
X_test = np.array(x_test)
Y_test = np.array(y_test)

print(X_train.shape)


# 在这里我们可以发现一张图片的形状为 (64, 64, 3)

# In[4]:


num_train_data = Y_train.shape[1]
num_test_data = Y_test.shape[1]
size_of_picture = X_train.shape[1] 


# In[15]:


# 查看猫的图片具体长什么样子的
import matplotlib.pyplot as plt
index = (25, 26, 27, 28)
plt.subplots(figsize=(20, 10))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(x_train[index[i]])


# 为了后面方便处理图片，我们将维度为(64, 64, 3)的数组重新构造为 (64x64x3, 1)的数组

# In[23]:


x_train_flatten = x_train.reshape(x_train.shape[0], -1).T

x_test_flatten = x_test.reshape(x_test.shape[0], -1).T


# In[24]:


print(x_train_flatten.shape)

print(x_test_flatten.shape)


# 接下来这步十分重要
# 为了表示彩色图像，那么就必须为每个像素指定红色，绿色和蓝色通道，像素值实际上就是从0到255范围内的三个数字的向量，这样让标准化的数据位于[0,1]之间，方便我们下一步的处理

# In[25]:


plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[26]:


X_train = x_train_flatten / 255.0
X_test = x_test_flatten / 255.0


# # 下面开始构建logistic的梯度下降法

# In[27]:


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# In[28]:


def initialize_w_b(dim):
    w = np.zeros(shape=(dim,1))
    
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int)) # 在这里不写这个也是可以的，但是我们为了规范代码，避免不必要发生的错误，我们还是严谨一点
    
    return (w, b)


# In[29]:


def propagate(w, b, X, Y):
    
    m = X.shape[1]
    # 正向传播
    z  = sigmoid(np.dot(w.T, X) + b)
    # 成本函数
    cost = (-1) * np.sum(Y * np.log(z) + (1 - Y) * (np.log(1 - z)))
    
    # 反向传播
    dw = (1 / m) * np.dot(X, (z- Y).T)
    db = (1/ m) * np.sum(z -  Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {
        'dw':dw,
        'db':db
    }
    return (grads, cost)


# In[30]:





# In[31]:


def  optimizer(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    
    for i in range (num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads['dw']
        db = grads['db']
        # 更新w，b
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
            
        if (print_cost) and (i % 100 == 0):
            print("迭代次数：i%，误差值：%f" % (i, cost))
            
    params = {
        'w':w,
        'b':b
    }
    grads ={
        'dw':dw,
        'db':db
    }
    return (params, grads, costs)


# In[32]:





# In[33]:


def predict(w, b, X):
    m  = X.shape[1] #图片的数量
    Y_prediction = np.zeros((1,m)) 
    w = w.reshape(X.shape[0],1)
    
    #计预测猫在图片中出现的概率
    z = sigmoid(np.dot(w.T , X) + b)
    for i in range(z.shape[1]):
        #将概率z [0，i]转换为实际预测p [0，i]
        Y_prediction[0,i] = 1 if z[0,i] > 0.5 else 0
    #使用断言
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction


# In[34]:


def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
    w , b = initialize_w_b(X_train.shape[0])
    
    parameters , grads , costs = optimizer(w , b , X_train , Y_train,num_iterations , learning_rate)
    
    #从字典“参数”中检索参数w和b
    w , b = parameters["w"] , parameters["b"]
    
    #预测测试/训练集的例子
    Y_prediction_test = predict(w , b, X_test)
    Y_prediction_train = predict(w , b, X_train)
    
    #打印训练后的准确性
    print("训练集准确性："  , format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100) ,"%")
    print("测试集准确性："  , format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) ,"%")
    
    stored = {
            "costs" : costs,
            "Y_prediction_test" : Y_prediction_test,
            "Y_prediciton_train" : Y_prediction_train,
            "w" : w,
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations }
    return stored


# In[37]:


d = model(X_train, y_train, X_test, y_test, num_iterations=2000, learning_rate=0.01)


# In[38]:


costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations ( hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# In[ ]:




