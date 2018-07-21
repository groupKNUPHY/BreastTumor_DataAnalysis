import tensorflow as tf
import numpy as np


filename_queue = tf.train.string_input_producer(['tumordata_anal.csv'],shuffle=False,name='filename_queue')

reader = tf.TextLineReader()
key,value = reader.read(filename_queue)



record_defaults=[[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]
#record_defaults = [[0.]*32]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[1:2],xy[1:2]],batch_size=10)

# Multivariable sample data
# y must be binary form

X = tf.placeholder(tf.float32, shape=[None,1])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([1,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

# hypothesis using sigmoid function

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

# New cost function

train = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(cost)

#Summrize train step by one line. Yes!

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))

# cast method makes True of False to dtype(=float)
# calculate prediction and accuracy 
# if the prediction and Y is same, equal method return True and cast method turn it into float

with tf.Session() as sess:

# It is same with sess = tf.Session() but more safety

	sess.run(tf.global_variables_initializer())
	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)


	for step in range(10001):
		x_batch, y_batch = sess.run([train_x_batch, train_y_batch])	
		 
		
		cost_val ,_ = sess.run([cost,train],feed_dict={X : x_batch, Y : y_batch})
		if step % 200 == 0:
			print(step, cost_val)
# Launch graph
	
	h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X : x_batch, Y : y_batch})
	print("hypothesis :", h)
	print("Your Tumor(1:die 0:alive) :", c)
	print("Accuracy :", a)
	print(x_batch)

	coord.request_stop()
	coord.join(threads)


