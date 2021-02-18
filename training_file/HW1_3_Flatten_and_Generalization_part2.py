import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# data preparation
data = input_data.read_data_sets('data/MNIST/', one_hot=True);
train_num = data.train.num_examples
valid_num = data.validation.num_examples
test_num = data.test.num_examples
img_flatten = 784
img_size = 28
num_classes = 10

# model 1-bath size
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu);
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2);
flat1 = tf.layers.flatten(pool1);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=0.001);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))
train_loss_list1 = []
train_acc_list1 = []
test_loss_list1 = []
test_acc_list1 = []
sens_list1 = []
session = tf.Session()
session.run(tf.global_variables_initializer())

BATCH_SIZE = [4,16,64,256,512,1024,2048,4096]
for i in range(len(BATCH_SIZE)):
    for j in range(int(data.train.num_examples/BATCH_SIZE[i])):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE[i])
        session.run(train_op, feed_dict={x: x_batch,y: y_true_batch})
    train_loss, train_acc = session.run([loss,acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list1.append(train_loss)
    train_acc_list1.append(train_acc)
    test_loss, test_acc, sens = session.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list1.append(test_loss)
    test_acc_list1.append(test_acc)
    sens_list1.append(sens)
    msg = "Batch Size: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
    print(msg.format(BATCH_SIZE[i], train_loss, train_acc, test_loss, test_acc, sens))

fig,axs=plt.subplots(1,2)
fig.set_figwidth(12)
fig.set_tight_layout('tight')
axs[0].plot(BATCH_SIZE,sens_list1,'r')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity')
axs[0].yaxis.label.set_color('red')
axs[0].set_xlabel('Batch Size(log scale)',size=10)
axs[0].legend(['Sensitivity'],loc=6)
axs1 = axs[0].twinx()
axs1.plot(BATCH_SIZE, train_loss_list1,'b')
axs1.plot(BATCH_SIZE, test_loss_list1,'b--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss')
axs1.yaxis.label.set_color('blue')
axs1.set_xlabel('Batch Size(log scale)')
axs1.legend(['Train','Test'],loc=4)
axs1.set_title('Loss vs Batch Size')

axs[1].plot(BATCH_SIZE,sens_list1,'r')
axs[1].set_xscale('log')
axs[1].set_ylabel('Sensitivity')
axs[1].yaxis.label.set_color('red')
axs[1].set_xlabel('Batch Size(log scale)',size=10)
axs[1].legend(['Sensitivity'],loc=6)
axs2 = axs[1].twinx()
axs2.plot(BATCH_SIZE, train_acc_list1,'b')
axs2.plot(BATCH_SIZE, test_acc_list1,'b--')
axs2.set_ylabel('Accuracy')
axs2.yaxis.label.set_color('blue')
axs2.set_xlabel('Batch Size(log scale)')
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Batch Size')

# model 1-learning rate


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)
learning_rate = tf.placeholder(tf.float32)

conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu);
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2);
flat1 = tf.layers.flatten(pool1);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))

train_loss_list2 = []
train_acc_list2 = []
test_loss_list2 = []
test_acc_list2 = []
sens_list2 = []
init = tf.global_variables_initializer()

lr_list = [0.05,0.01,0.005,0.001,0.0005,0.0001]
init = tf.global_variables_initializer()
Batchsize = 128

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(lr_list)):
        for j in range(data.train.num_examples//Batchsize):
            x_batch, y_batch = data.train.next_batch(Batchsize)
            sess.run(train_op, feed_dict={x: x_batch,y: y_batch,learning_rate:lr_list[i]})
        train_loss, train_acc = sess.run([loss,acc_op],feed_dict={x:x_batch,y:y_batch})
        train_loss_list2.append(train_loss)
        train_acc_list2.append(train_acc)
        test_loss, test_acc, sens = sess.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
        test_loss_list2.append(test_loss)
        test_acc_list2.append(test_acc)
        sens_list2.append(sens)
        msg = "Learn rate: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
        print(msg.format(lr_list[i], train_loss, train_acc, test_loss, test_acc, sens))

fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(lr_list,sens_list2,'r')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity')
axs[0].yaxis.label.set_color('red')
axs[0].set_xlabel('Learning Rate (log scale)')
axs[0].legend(['Sensitivity'])
axs1 = axs[0].twinx()
axs1.plot(lr_list, train_loss_list2,'b')
axs1.plot(lr_list, test_loss_list2,'b--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss')
axs1.yaxis.label.set_color('blue')
axs1.set_xlabel('Learning Rate(log scale)')
axs1.legend(['Train','Test'])
axs1.set_title('Loss vs Learning Rate')

axs[1].plot(lr_list,sens_list2,'r')
axs[1].set_xscale('log')
axs[1].set_ylabel('Sensitivity')
axs[1].yaxis.label.set_color('red')
axs[1].set_xlabel('Learning Rate (log scale)')
axs[1].legend(['Sensitivity'])
axs2 = axs[1].twinx()
axs2.plot(lr_list, train_acc_list2,'b')
axs2.plot(lr_list, test_acc_list2,'b--')
axs2.set_ylabel('Accuracy')
axs2.set_xlabel('Learning Rate(log scale)')
axs2.yaxis.label.set_color('blue')
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Learning Rate')


# model 2-batch size
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu);
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2);
conv2 = tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu);
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2);
flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=0.001);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))
train_loss_list1 = []
train_acc_list1 = []
test_loss_list1 = []
test_acc_list1 = []
sens_list1 = []
session = tf.Session()
session.run(tf.global_variables_initializer())

BATCH_SIZE = [4,16,64,256,512,1024,2048,4096]
for i in range(len(BATCH_SIZE)):
    for j in range(int(data.train.num_examples/BATCH_SIZE[i])):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE[i])
        session.run(train_op, feed_dict={x: x_batch,y: y_true_batch})
    train_loss, train_acc = session.run([loss,acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list1.append(train_loss)
    train_acc_list1.append(train_acc)
    test_loss, test_acc, sens = session.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list1.append(test_loss)
    test_acc_list1.append(test_acc)
    sens_list1.append(sens)
    msg = "Batch Size: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
    print(msg.format(BATCH_SIZE[i], train_loss, train_acc, test_loss, test_acc, sens))

fig,axs=plt.subplots(1,2)
fig.set_figwidth(12)
fig.set_tight_layout('tight')
axs[0].plot(BATCH_SIZE,sens_list1,'r')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity')
axs[0].yaxis.label.set_color('red')
axs[0].set_xlabel('Batch Size(log scale)',size=10)
axs[0].legend(['Sensitivity'])
axs1 = axs[0].twinx()
axs1.plot(BATCH_SIZE, train_loss_list1,'b')
axs1.plot(BATCH_SIZE, test_loss_list1,'b--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss')
axs1.yaxis.label.set_color('blue')
axs1.set_xlabel('Batch Size(log scale)')
axs1.legend(['Train','Test'],loc=4)
axs1.set_title('Loss vs Batch Size')

axs[1].plot(BATCH_SIZE,sens_list1,'r')
axs[1].set_xscale('log')
axs[1].set_ylabel('Sensitivity')
axs[1].yaxis.label.set_color('red')
axs[1].set_xlabel('Batch Size(log scale)',size=10)
axs[1].legend(['Sensitivity'],loc=6)
axs2 = axs[1].twinx()
axs2.plot(BATCH_SIZE, train_acc_list1,'b')
axs2.plot(BATCH_SIZE, test_acc_list1,'b--')
axs2.set_ylabel('Accuracy')
axs2.yaxis.label.set_color('blue')
axs2.set_xlabel('Batch Size(log scale)')
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Batch Size')


# model 2-learning rate
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)
learning_rate = tf.placeholder(tf.float32)

conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu);
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2);
conv2 = tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu);
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2);
flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))
train_loss_list2 = []
train_acc_list2 = []
test_loss_list2 = []
test_acc_list2 = []
sens_list2 = []
session = tf.Session()
session.run(tf.global_variables_initializer())

lr_list = [0.05,0.01,0.005,0.001,0.0005,0.0001]
BATCH_SIZE = 64
for i in range(len(lr_list)):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(train_op, feed_dict={x: x_batch,y: y_true_batch,learning_rate:lr_list[i]})
    train_loss, train_acc = session.run([loss,acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list2.append(train_loss)
    train_acc_list2.append(train_acc)
    test_loss, test_acc, sens = session.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list2.append(test_loss)
    test_acc_list2.append(test_acc)
    sens_list2.append(sens)
    msg = "Learn rate: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
    print(msg.format(lr_list[i], train_loss, train_acc, test_loss, test_acc, sens))
fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(lr_list,sens_list2,'r')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity')
axs[0].yaxis.label.set_color('red')
axs[0].set_xlabel('Learning Rate (log scale)')
axs[0].legend(['Sensitivity'])
axs1 = axs[0].twinx()
axs1.plot(lr_list, train_loss_list2,'b')
axs1.plot(lr_list, test_loss_list2,'b--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss')
axs1.yaxis.label.set_color('blue')
axs1.set_xlabel('Learning Rate(log scale)')
axs1.legend(['Train','Test'])
axs1.set_title('Loss vs Learning Rate')

axs[1].plot(lr_list,sens_list2,'r')
axs[1].set_xscale('log')
axs[1].set_ylabel(['Sensitivity'])
axs[1].yaxis.label.set_color('red')
axs[1].set_xlabel('Learning Rate (log scale)')
axs[1].legend(['Sensitivity'])
axs2 = axs[1].twinx()
axs2.plot(lr_list, train_acc_list2,'b')
axs2.plot(lr_list, test_acc_list2,'b--')
axs2.set_ylabel('Accuracy')
axs2.set_xlabel('Learning Rate(log scale)')
axs2.yaxis.label.set_color('blue')
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Learning Rate')
# model 3_batch size
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

flat1 = tf.layers.flatten(inputs=input_x)
h1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.relu);
h2 = tf.layers.dense(inputs=h1,units=256,activation=tf.nn.relu);
h3 = tf.layers.dense(inputs=h1,units=128,activation=tf.nn.relu);
h4 = tf.layers.dense(inputs=h1,units=64,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=h4,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=0.001);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))
train_loss_list1 = []
train_acc_list1 = []
test_loss_list1 = []
test_acc_list1 = []
sens_list1 = []
session = tf.Session()
session.run(tf.global_variables_initializer())

BATCH_SIZE = [4,16,64,256,512,1024,2048,4096]
for i in range(len(BATCH_SIZE)):
    for j in range(int(data.train.num_examples/BATCH_SIZE[i])):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE[i])
        session.run(train_op, feed_dict={x: x_batch,y: y_true_batch})
    train_loss, train_acc = session.run([loss,acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list1.append(train_loss)
    train_acc_list1.append(train_acc)
    test_loss, test_acc, sens = session.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list1.append(test_loss)
    test_acc_list1.append(test_acc)
    sens_list1.append(sens)
    msg = "Batch Size: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
    print(msg.format(BATCH_SIZE[i], train_loss, train_acc, test_loss, test_acc, sens))
fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(BATCH_SIZE,sens_list1,'r')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity')
axs[0].yaxis.label.set_color('red')
axs[0].set_xlabel('Batch Size(log scale)')
axs[0].legend(['Sensitivity'])
axs1 = axs[0].twinx()
axs1.plot(BATCH_SIZE, train_loss_list1,'b')
axs1.plot(BATCH_SIZE, test_loss_list1,'b--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss')
axs1.yaxis.label.set_color('blue')
axs1.legend(['Train','Test'])
axs1.set_title('Loss vs Batch Size')


axs[1].plot(BATCH_SIZE,sens_list1,'r')
axs[1].set_xscale('log')
axs[1].set_ylabel('Sensitivity')
axs[1].yaxis.label.set_color('red')
axs[1].set_xlabel('Batch Size(log scale)')
axs[1].legend(['Sensitivity'])
axs2 = axs[1].twinx()
axs2.plot(BATCH_SIZE, train_acc_list1,'b')
axs2.plot(BATCH_SIZE, test_acc_list1,'b--')
axs2.set_ylabel('Accuracy')
axs2.yaxis.label.set_color('blue')
axs2.set_xlabel('Batch Size(log scale)')
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Batch Size')

# model 3-learning rate
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)
learning_rate = tf.placeholder(tf.float32)

flat1 = tf.layers.flatten(inputs=input_x)
h1 = tf.layers.dense(inputs=flat1,units=128,activation=tf.nn.relu);
h2 = tf.layers.dense(inputs=h1,units=256,activation=tf.nn.relu);
h3 = tf.layers.dense(inputs=h1,units=128,activation=tf.nn.relu);
h4 = tf.layers.dense(inputs=h1,units=64,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=h4,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))
train_loss_list2 = []
train_acc_list2 = []
test_loss_list2 = []
test_acc_list2 = []
sens_list2 = []
session = tf.Session()
session.run(tf.global_variables_initializer())

lr_list = [0.05,0.01,0.005,0.001,0.0005,0.0001]
BATCH_SIZE = 64
for i in range(len(lr_list)):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(train_op, feed_dict={x: x_batch,y: y_true_batch,learning_rate:lr_list[i]})
    train_loss, train_acc = session.run([loss,acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list2.append(train_loss)
    train_acc_list2.append(train_acc)
    test_loss, test_acc, sens = session.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list2.append(test_loss)
    test_acc_list2.append(test_acc)
    sens_list2.append(sens)
    msg = "Learn rate: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
    print(msg.format(lr_list[i], train_loss, train_acc, test_loss, test_acc, sens))
fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(lr_list,sens_list2,'r')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity')
axs[0].yaxis.label.set_color('red')
axs[0].set_xlabel('Learning Rate (log scale)')
axs[0].legend(['Sensitivity'])
axs1 = axs[0].twinx()
axs1.plot(lr_list, train_loss_list2,'b')
axs1.plot(lr_list, test_loss_list2,'b--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss')
axs1.yaxis.label.set_color('blue')
axs1.set_xlabel('Learning Rate(log scale)')
axs1.legend(['Train','Test'])
axs1.set_title('Loss vs Learning Rate')

axs[1].plot(lr_list,sens_list2,'r')
axs[1].set_xscale('log')
axs[1].set_ylabel(['Sensitivity'])
axs[1].yaxis.label.set_color('red')
axs[1].set_xlabel('Learning Rate (log scale)')
axs[1].legend(['Sensitivity'])
axs2 = axs[1].twinx()
axs2.plot(lr_list, train_acc_list2,'b')
axs2.plot(lr_list, test_acc_list2,'b--')
axs2.set_ylabel('Accuracy')
axs2.set_xlabel('Learning Rate(log scale)')
axs2.yaxis.label.set_color('blue')
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Learning Rate')

# model 4-batch size
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu);
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2);
flat1 = tf.layers.flatten(pool1);
fc1 = tf.layers.dense(inputs=flat1,units=64,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=0.001);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))
train_loss_list1 = []
train_acc_list1 = []
test_loss_list1 = []
test_acc_list1 = []
sens_list1 = []
session = tf.Session()
session.run(tf.global_variables_initializer())

BATCH_SIZE = [4,16,64,256,512,1024,2048,4096]
for i in range(len(BATCH_SIZE)):
    for j in range(int(data.train.num_examples/BATCH_SIZE[i])):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE[i])
        session.run(train_op, feed_dict={x: x_batch,y: y_true_batch})
    train_loss, train_acc = session.run([loss,acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list1.append(train_loss)
    train_acc_list1.append(train_acc)
    test_loss, test_acc, sens = session.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list1.append(test_loss)
    test_acc_list1.append(test_acc)
    sens_list1.append(sens)
    msg = "Batch Size: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
    print(msg.format(BATCH_SIZE[i], train_loss, train_acc, test_loss, test_acc, sens))
fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(BATCH_SIZE,sens_list1,'r')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity')
axs[0].yaxis.label.set_color('red')
axs[0].set_xlabel('Batch Size(log scale)')
axs[0].legend(['Sensitivity'])
axs1 = axs[0].twinx()
axs1.plot(BATCH_SIZE, train_loss_list1,'b')
axs1.plot(BATCH_SIZE, test_loss_list1,'b--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss')
axs1.yaxis.label.set_color('blue')
axs1.legend(['Train','Test'])
axs1.set_title('Loss vs Batch Size')


axs[1].plot(BATCH_SIZE,sens_list1,'r')
axs[1].set_xscale('log')
axs[1].set_ylabel('Sensitivity')
axs[1].yaxis.label.set_color('red')
axs[1].set_xlabel('Batch Size(log scale)')
axs[1].legend(['Sensitivity'])
axs2 = axs[1].twinx()
axs2.plot(BATCH_SIZE, train_acc_list1,'b')
axs2.plot(BATCH_SIZE, test_acc_list1,'b--')
axs2.set_ylabel('Accuracy')
axs2.yaxis.label.set_color('blue')
axs2.set_xlabel('Batch Size(log scale)')
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Batch Size')

#model 4 -learning rate
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)
learning_rate = tf.placeholder(tf.float32)

conv1 = tf.layers.conv2d(inputs=input_x,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu);
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2);
flat1 = tf.layers.flatten(pool1);
fc1 = tf.layers.dense(inputs=flat1,units=64,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))
train_loss_list2 = []
train_acc_list2 = []
test_loss_list2 = []
test_acc_list2 = []
sens_list2 = []
session = tf.Session()
session.run(tf.global_variables_initializer())

lr_list = [0.05,0.01,0.005,0.001,0.0005,0.0001]
BATCH_SIZE = 64
for i in range(len(lr_list)):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(train_op, feed_dict={x: x_batch,y: y_true_batch,learning_rate:lr_list[i]})
    train_loss, train_acc = session.run([loss,acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list2.append(train_loss)
    train_acc_list2.append(train_acc)
    test_loss, test_acc, sens = session.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list2.append(test_loss)
    test_acc_list2.append(test_acc)
    sens_list2.append(sens)
    msg = "Learn rate: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
    print(msg.format(lr_list[i], train_loss, train_acc, test_loss, test_acc, sens))
fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(lr_list,sens_list2,'r')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity')
axs[0].yaxis.label.set_color('red')
axs[0].set_xlabel('Learning Rate (log scale)')
axs[0].legend(['Sensitivity'])
axs1 = axs[0].twinx()
axs1.plot(lr_list, train_loss_list2,'b')
axs1.plot(lr_list, test_loss_list2,'b--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss')
axs1.yaxis.label.set_color('blue')
axs1.set_xlabel('Learning Rate(log scale)')
axs1.legend(['Train','Test'])
axs1.set_title('Loss vs Learning Rate')

axs[1].plot(lr_list,sens_list2,'r')
axs[1].set_xscale('log')
axs[1].set_ylabel(['Sensitivity'])
axs[1].yaxis.label.set_color('red')
axs[1].set_xlabel('Learning Rate (log scale)')
axs[1].legend(['Sensitivity'])
axs2 = axs[1].twinx()
axs2.plot(lr_list, train_acc_list2,'b')
axs2.plot(lr_list, test_acc_list2,'b--')
axs2.set_ylabel('Accuracy')
axs2.set_xlabel('Learning Rate(log scale)')
axs2.yaxis.label.set_color('blue')
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Learning Rate')

#model 5 -batch size
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

conv1 = tf.layers.conv2d(inputs=input_x,filters=1,kernel_size=3,padding="same",activation=tf.nn.relu);
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2);
conv2 = tf.layers.conv2d(inputs=pool1,filters=1,kernel_size=3,padding="same",activation=tf.nn.relu);
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=4,strides=4);
flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=10,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=0.001);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))
train_loss_list1 = []
train_acc_list1 = []
test_loss_list1 = []
test_acc_list1 = []
sens_list1 = []
session = tf.Session()
session.run(tf.global_variables_initializer())

BATCH_SIZE = [4,16,64,256,512,1024,2048,4096]
for i in range(len(BATCH_SIZE)):
    for j in range(int(data.train.num_examples/BATCH_SIZE[i])):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE[i])
        session.run(train_op, feed_dict={x: x_batch,y: y_true_batch})
    train_loss, train_acc = session.run([loss,acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list1.append(train_loss)
    train_acc_list1.append(train_acc)
    test_loss, test_acc, sens = session.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list1.append(test_loss)
    test_acc_list1.append(test_acc)
    sens_list1.append(sens)
    msg = "Batch Size: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
    print(msg.format(BATCH_SIZE[i], train_loss, train_acc, test_loss, test_acc, sens))

fig,axs=plt.subplots(1,2)
fig.set_figwidth(12)
fig.set_tight_layout('tight')
axs[0].plot(BATCH_SIZE,sens_list1,'r')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity')
axs[0].yaxis.label.set_color('red')
axs[0].set_xlabel('Batch Size(log scale)',size=10)
axs[0].legend(['Sensitivity'],loc=6)
axs1 = axs[0].twinx()
axs1.plot(BATCH_SIZE, train_loss_list1,'b')
axs1.plot(BATCH_SIZE, test_loss_list1,'b--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss')
axs1.yaxis.label.set_color('blue')
axs1.set_xlabel('Batch Size(log scale)')
axs1.legend(['Train','Test'],loc=4)
axs1.set_title('Loss vs Batch Size')

axs[1].plot(BATCH_SIZE,sens_list1,'r')
axs[1].set_xscale('log')
axs[1].set_ylabel('Sensitivity')
axs[1].yaxis.label.set_color('red')
axs[1].set_xlabel('Batch Size(log scale)',size=10)
axs[1].legend(['Sensitivity'],loc=6)
axs2 = axs[1].twinx()
axs2.plot(BATCH_SIZE, train_acc_list1,'b')
axs2.plot(BATCH_SIZE, test_acc_list1,'b--')
axs2.set_ylabel('Accuracy')
axs2.yaxis.label.set_color('blue')
axs2.set_xlabel('Batch Size(log scale)')
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Batch Size')

# model 5-learning rate
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)
learning_rate = tf.placeholder(tf.float32)

conv1 = tf.layers.conv2d(inputs=input_x,filters=1,kernel_size=3,padding="same",activation=tf.nn.relu);
pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=2,strides=2);
conv2 = tf.layers.conv2d(inputs=pool1,filters=1,kernel_size=3,padding="same",activation=tf.nn.relu);
pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=4,strides=4);
flat1 = tf.layers.flatten(pool2);
fc1 = tf.layers.dense(inputs=flat1,units=10,activation=tf.nn.relu);
logits = tf.layers.dense(inputs=fc1,units=num_classes,activation=None);
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits);
loss = tf.reduce_mean(cross_entropy);

# Accuracy
softmax = tf.nn.softmax(logits=logits);
pred_op = tf.argmax(softmax,dimension=1);
acc_op = tf.reduce_mean(tf.cast(tf.equal(pred_op, y_cls), tf.float32));
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate);
train_op = optimizer.minimize(loss);
sens_op = tf.norm(tf.gradients(loss,input_x))
train_loss_list2 = []
train_acc_list2 = []
test_loss_list2 = []
test_acc_list2 = []
sens_list2 = []
session = tf.Session()
session.run(tf.global_variables_initializer())

lr_list = [0.05,0.01,0.005,0.001,0.0005,0.0001]
BATCH_SIZE = 64
for i in range(len(lr_list)):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(train_op, feed_dict={x: x_batch,y: y_true_batch,learning_rate:lr_list[i]})
    train_loss, train_acc = session.run([loss,acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list2.append(train_loss)
    train_acc_list2.append(train_acc)
    test_loss, test_acc, sens = session.run([loss,acc_op,sens_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list2.append(test_loss)
    test_acc_list2.append(test_acc)
    sens_list2.append(sens)
    msg = "Learn rate: {0:>4}, Training Loss: {1:>1.4}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.4}, Test Accuracy: {4:>6.1%}, Sensitivity: {5:>1.4}"
    print(msg.format(lr_list[i], train_loss, train_acc, test_loss, test_acc, sens))
fig,axs=plt.subplots(1,2)
fig.set_figwidth(15)
fig.set_tight_layout('tight')
axs[0].plot(lr_list,sens_list2,'r')
axs[0].set_xscale('log')
axs[0].set_ylabel('Sensitivity')
axs[0].yaxis.label.set_color('red')
axs[0].set_xlabel('Learning Rate (log scale)')
axs[0].legend(['Sensitivity'])
axs1 = axs[0].twinx()
axs1.plot(lr_list, train_loss_list2,'b')
axs1.plot(lr_list, test_loss_list2,'b--')
axs1.set_xscale('log')
axs1.set_ylabel('Loss')
axs1.yaxis.label.set_color('blue')
axs1.set_xlabel('Learning Rate(log scale)')
axs1.legend(['Train','Test'])
axs1.set_title('Loss vs Learning Rate')

axs[1].plot(lr_list,sens_list2,'r')
axs[1].set_xscale('log')
axs[1].set_ylabel(['Sensitivity'])
axs[1].yaxis.label.set_color('red')
axs[1].set_xlabel('Learning Rate (log scale)')
axs[1].legend(['Sensitivity'])
axs2 = axs[1].twinx()
axs2.plot(lr_list, train_acc_list2,'b')
axs2.plot(lr_list, test_acc_list2,'b--')
axs2.set_ylabel('Accuracy')
axs2.set_xlabel('Learning Rate(log scale)')
axs2.yaxis.label.set_color('blue')
axs2.legend(['Train','Test'])
axs2.set_title('Accuracy vs Learning Rate')