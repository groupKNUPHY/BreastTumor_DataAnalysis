import tensorflow as tf
import numpy as np

def MinMaxScaler(data):
	numerator = data - np.min(data,0)
	denominator = np.max(data,0)-np.min(data,0)
	return numerator/(denominator + 1e-7)

#filename_queue = tf.train.string_input_producer(['data.csv'],shuffle=False,name='filename_queue')

#reader = tf.TextLineReader()
#key,value = reader.read(filename_queue)

#record_defaults=[[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]
#record_defaults = [[0.]*32]

xy=np.loadtxt('/home/xoqhdgh/tensor/BreastTumor_DataAnalysis/data.csv',delimiter=',',dtype = np.float32)
xy1=xy[:,2:32]
xy2=xy[:,1:2]
#xy = tf.decode_csv(value, record_defaults=record_defaults)
xy1 = np.array(xy1)

xy1 = MinMaxScaler(xy1)

x_data = xy1[:-10,:]
y_data = xy2[:-10,[0]]
x_test = xy1[-10:,:]
y_test = xy2[-10:,[0]]
#train_x_batch, train_y_batch = tf.train.batch([xy[2:32],xy[1:2]],batch_size=563)

# Multivariable sample data
# y must be binary form

X = tf.placeholder(tf.float32, shape=[None,30])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([30,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

# hypothesis using sigmoid function

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

# New cost function

train = tf.train.GradientDescentOptimizer(learning_rate=0.04).minimize(cost)

#Summrize train step by one line. Yes!

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))

# cast method makes True of False to dtype(=float)
# calculate prediction and accuracy 
# if the prediction and Y is same, equal method return True and cast method turn it into float

with tf.Session() as sess:

# It is same with sess = tf.Session() but more safety

	sess.run(tf.global_variables_initializer())
	
	#coord = tf.train.Coordinator()
	#threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	for step in range(10001):
		#x_batch, y_batch = sess.run([train_x_batch, train_y_batch])	
		
		cost_val ,hy_val,_ = sess.run([cost,hypothesis,train],feed_dict={X : x_data, Y : y_data})
		if step % 200 == 0:
			print(step)
			print(cost_val)
			print("="*20)
# Launch graph
	
	h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X : x_test, Y : y_test})
	print("\nhypothesis :",h,"\nYour Tumor(1:die 0:alive) :",c,"\nAccuracy :",a)
	#print(x_data)

	#coord.request_stop()
	#coord.join(threads)

