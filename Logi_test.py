import tensorflow as tf
import numpy as np

xy = np.loadtxt('sample.csv',delimiter=',',dtype=np.float32)

x_data = xy[:-6,2:]
y_data = xy[:-6,1]

test_x_data = xy[-6:,2:]
test_y_data = xy[-6:,1]



# Multivariable sample data
# y must be binary form

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

# hypothesis using sigmoid function

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

# New cost function

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#Summrize train step by one line. Yes!

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))

# cast method makes True of False to dtype(=float)
# calculate prediction and accuracy 
# if the prediction and Y is same, equal method return True and cast method turn it into float

with tf.Session() as sess:

# It is same with sess = tf.Session() but more safety

	sess.run(tf.global_variables_initializer())
	
	for step in range(10001):
		cost_val ,_ = sess.run([cost,train],feed_dict={X : x_data, Y : y_data})
		if step % 200 == 0:
			print(step, cost_val)
# Launch graph
	
	h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X : x_data, Y : y_data})
	print("hypothesis :", h)
	print("Correct(Y) :", c)
	print("Accuracy :", a)

	print("Your result(pass:  or not) is: ")
	result = sess.run(predicted,feed_dict={X:[[5,1]]})
	if 1.0 == result[0][0]:
		print("PASS")
	else:
		print("DROP")

