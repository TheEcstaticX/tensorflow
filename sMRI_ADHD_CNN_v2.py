#!/usr/bin/env python

"""Create and train a convolutional neural network model on 3D structural human MRI data as a classifier for ADHD and non-ADHD brains"""

import tensorflow as tf
import numpy as np
import nibabel as nib
import os
import fnmatch
from timeit import default_timer as timer

__author__ = "Nicholas Nguyen, Eric Barnett"
__credits__ = ["Nicholas Nguyen", "Eric Barnett", "Stephen Faraone", "Ioana Coman", "Wendy Kates"]

__version__ = "0.0.3"
__maintainer__ = "Nicholas Nguyen"
__email__ = "nguyenni@upstate.edu"
__status__ = "Prototype"


start = timer()
## Objective 1 : Create data input pipeline

# Load all .nii files in the directory.
# Each .nii file is loaded using the NiBabel library, and the 3D array
# containing intensity values is acquired. The raw arrays are flipped
# around the x direction so the brain is in the correct orientation.
print("Begin image loading...")
data = []
for file in sorted(os.listdir('.')):
	if fnmatch.fnmatch(file, '*.nii'):
		img = nib.load(file)
		temp = np.flipud(img.get_data())
		data.append(temp)
print("Image loading complete")

print("Number of images loaded: " + str(len(data)))
data = np.asarray(data)

# Load data labels from phenotype file: 'adhd200_phenotypics.csv'
# Convert ADHD subtype labels to single 'case' label
print("Begin loading phenotype labels...")
phenotype = np.genfromtxt('adhd200_mini_phenotypics.csv', delimiter = ',', skip_header = 1)
labels = []
for i in xrange(0, len(phenotype)):
	labels.append(phenotype[i,2:4])
print("Phenotype label loading complete")

print("Number of phenotypes loaded: " + str(len(labels)))
labels = np.asarray(labels)



## Objective 2 : Create convolutional network

# %%
def conv3d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input image
    x = tf.reshape(x, shape=[-1, 121, 145, 121, 1])
    # Convolution Layer
    conv2 = conv3d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    #conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    #conv2 = conv3d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    #conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv1 output to fit fully connected layer input
    conv2shape = conv2.get_shape().as_list()
    fc1 = tf.reshape(conv2, [-1, conv2shape[1]*conv2shape[2]*conv2shape[3]*conv2shape[4]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Training Parameters
learning_rate = 0.01
training_iters = 2
batch_size = 1
display_step = 1

# Network Parameters
n_input = [None, 121, 145, 121] # sMRI data input (img shape: 121 x 145 x 121)
n_classes = 1 # sMRI total classes (case v. control
dropout = 0.5 # Dropout, probability to keep units

# Store layers weight & bias
weights = {
    # 3x3 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 3, 1, 4])),
    # 3x3 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 3, 4, 4])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8491780, 16])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([16, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([4])),
    'bc2': tf.Variable(tf.random_normal([4])),
    'bd1': tf.Variable(tf.random_normal([16])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Input placeholders:
x = tf.placeholder(tf.float32, n_input)
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    print "Begin model training..."

    for i in range(training_iters):
        # Define batches
        training_batch = zip(range(0, len(data), batch_size),
                             range(batch_size, len(data) + 1, batch_size))
        step = 1
        for start, end in training_batch:
            sess.run(optimizer, feed_dict={x: data[start:end], y: labels[start:end],
        	                               keep_prob: dropout})
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: data[start:end],
                                                              y: labels[start:end],
                                                              keep_prob: 1.})
            print "Iter " + str(i) + ", Batch " + str(step) + \
                  ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc)
            step += 1
    print "Model training complete"

end = timer()
print(end - start)