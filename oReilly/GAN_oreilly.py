""" GAN implementation with help from https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners """
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
# mnist contains data (images) with corresponding labels divided into training
# and validation set

sample_image = mnist.train.next_batch(1)[0]
print(sample_image.shape)   # (1, 784)
# => single row with 784 pixels
# => reshape to 28x28 images + visualize with PyPlot

sample_image = sample_image.reshape([28, 28])
imgplt = plt.imshow(sample_image, cmap='Greys')
plt.show()


# build discriminator #########################################################
#   two conv layers 5x5
#   two fully connected layers
def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # First convolutional and pool layers
        # => find 32 different 5x5 pixel features
        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # Second conv and pool layers
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)
        
        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1],
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4
        
        # d4 contains unscaled values
        return d4


# build generator #############################################################
#   upsample image using fractionally strided convolutions
#   stabilize training using batch_norm and ReLU
#   output layer is sigmoid activation to get only black/white pixels
def generator(z, batch_size, z_dim):
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
    g1 = tf.nn.relu(g1)
    
    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])
    
    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])
    
    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)
    
    # dimensions of g4: batch_size x 28 x 28 x 1
    return g4


# Generating a sample image ###################################################
z_dimensions = 100
# None keyword means that first dimensions of the shape can be determined at
# runtime => this enables us to use a variable batch_size (which gets specified)
# later
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

# generated_image_output will hold the output of the generator
generated_image_output = generator(z_placeholder, 1, z_dimensions)
# initialize random noise vector, arguments of np.random.normal:
#   (mean, stddev, shape of vector)
z_batch = np.random.normal(0, 1, [1, z_dimensions])

# sess.run() has two arguments:
#   first: "fetches" argument: value which should be computed
#   second: feed_dict: values feeded into a placeholder
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    generated_image = sess.run(generated_image_output,
                                feed_dict={z_placeholder : z_batch})
    generated_image = generated_image.reshape([28, 28])
    plt.imshow(generated_image, cmap='Greys')
    plt.show()


# Train the GAN ###############################################################
# Note: GANs have two loss functions: one for G and one for D
# Here we train G and D simultaneously
tf.reset_default_graph()
batch_size = 50

# z_placeholder is for feeding noise to the G
# None keyword is again for batch_size
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions],
                                name='z_placeholder')

# x_placeholder is for feeding images to the D
x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1],
                                name='x_placeholder')

# Gz holds the generated images
Gz = generator(z_placeholder, batch_size, z_dimensions)

# Dx will hold D prediction probabilities for the real MNIST images
Dx = discriminator(x_placeholder)

# Dg will hold discriminator prediction probabilities for generated images
Dg = discriminator(Gz, reuse_variables=True)

# loss for discriminator has two parts: #######################################
#   1. comparing Dx and 1 for real images
#   2. comparing Dg and 0 for fake images
# use tf.sigmoid_cross_entropy_with_logits(), because it operates on unscaled
# values, because we have no softmax/sigmoid layer at end of D
# tf.reduce_mean() takes the mean value of all of the components in the matrix
# returned by tf.sigmoid_cross_entropy_with_logits()
# => reduces loss to single scalar
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, 
                            labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, 
                            labels=tf.zeros_like(Dg)))


# loss for generator ##########################################################
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, 
                        labels=tf.ones_like(Dg)))


# define optimizers ###########################################################
# optimizer for G needs to only update the weights of G, not those of D
# => need 2 sets of variables
tvars = tf.trainable_variables()  # all trainable parameters

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

print([v.name for v in d_vars])
print([v.name for v in g_vars])

# use Adam optimizer, because it uses first and second order momentum
# as well as adaptive learning rate

# D is trained two times: first on fake images from G, then on real images
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake,
                            var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, 
                            var_list=d_vars)

# G optimizer
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()


sess = tf.Session()


# visualize with TensorBoard ##################################################
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)


# start the training ##########################################################
sess.run(tf.global_variables_initializer())

# pre-train D to give it a head start
print("Pre-training starts:")
for i in range(300):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake], 
                                            {x_placeholder: real_image_batch, z_placeholder: z_batch})
    
    if (i % 100 == 0):
        print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

print("Simultaneous training starts:")
# train G and D together
for i in range(100000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    z_batch = np.random.normal(9, 1, size=[batch_size, z_dimensions])
    
    # Train D on real and fake images
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                            {x_placeholder: real_image_batch, z_placeholder: z_batch})
    
    # Train G
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})
    
    # update TensorBoard every ten epochs
    if i % 10 == 0:
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
        writer.add_summary(summary, i)
    
    if i % 100 == 0:
        # show generated image every 100 epochs
        print("Iteration:", i, "at", datetime.datetime.now())
        z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
        generated_images = generator(z_placeholder, 1, z_dimensions)
        images = sess.run(generated_images, {z_placeholder: z_batch})
        plt.imshow(images[0].reshape([28, 28]),cmap='Greys')
        plt.show()
        
        # show D estimate
        im = images[0].reshape([1, 28, 28, 1])
        result = discriminator(x_placeholder)
        estimate = sess.run(result, {x_placeholder: im})
        print("Estimate:", estimate)


"""
    Practical problems (and ideas to correct them)
    1. Scenario: D overpowers G
        D always classifies generated images as 100% fake, problem: there is no
        gradient left to train the generator
        idea: dont use sigmoid-function as output for D -> output unscaled
    2. Scenario: Mode collapse
        G finds a weak spot in the D -> G generates many very similar images,
        regardless of z-input
        idea: strengthen the D, by eg adjusting learning rate or reconfiguring
        its layers
    
    Further "GAN hacks": https://github.com/soumith/ganhacks
"""