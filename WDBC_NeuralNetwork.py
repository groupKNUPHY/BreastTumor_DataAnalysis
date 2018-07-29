import tensorflow as tf
import numpy as np
import random
tf.set_random_seed(777)

def MinMaxScaler(data):
	numerator = data - np.min(data,0)
	denominator = np.max(data,0) - np.min(data,0)
	return numerator / (denominator+1e-7)


xy = np.loadtxt('tumordata_anal.csv',delimiter=',',dtype=np.float32)

train_x_data = xy[:398,2:]
train_x_data = MinMaxScaler(train_x_data )
train_y_data = xy[:398,1:2]

test_x_data = xy[398:,2:]
test_x_data = MinMaxScaler(test_x_data )
test_y_data = xy[398:,1:2]

print(train_x_data.shape)
print(test_x_data.shape)
print(xy.shape)

X = tf.placeholder(tf.float32, shape=[None,30])
Y = tf.placeholder(tf.float32, shape=[None,1])
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("layer1") as scope:
  
	W1 = tf.get_variable("W1", shape=[30, 30],
                       initializer=tf.contrib.layers.xavier_initializer())
	b1 = tf.Variable(tf.random_normal([30]))
	L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
	L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

with tf.name_scope("layer2") as scope:
  
	W2 = tf.get_variable("W2", shape=[30, 30],
	                   initializer=tf.contrib.layers.xavier_initializer())
	b2 = tf.Variable(tf.random_normal([30]))
	L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
	L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

with tf.name_scope("layer3") as scope:
  
	W3 = tf.get_variable("W3", shape=[30, 1],
	                   initializer=tf.contrib.layers.xavier_initializer())
	b3 = tf.Variable(tf.random_normal([1]))
	hypothesis = tf.nn.sigmoid(tf.matmul(L2,W3)+b3)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))

cost_sum=tf.summary.scalar('cost',cost)
accuracy_sum=tf.summary.scalar('accuracy',accuracy)
hypothesis_hist = tf.summary.histogram("hypothesis",hypothesis)

summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#total_epoch=6
total_epoch=11

writer=tf.summary.FileWriter("./log/drop")
writer.add_graph(sess.graph)

global_step=0
for epoch in range(total_epoch):
	
	avg_cost =0
	for i in range(569):     
		s,cost_val , _ = sess.run([summary,cost,train],feed_dict={X : train_x_data, Y : train_y_data,keep_prob:0.7})
		avg_cost +=cost_val/569
	
		if i%100==0:
			writer.add_summary(s,global_step=i)
		
	print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X : test_x_data, Y : test_y_data, keep_prob:1})
#print("hypothesis :", h)
#print("Correct(Y) :", c)
print("Accuracy :", a)
    

