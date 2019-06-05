"""
    OReilly Tutorial: Deep Convolutional Generative Adversarial Networks with TensorFlow
    Code: https://github.com/dmonn/dcgan-oreilly
    Article: https://www.oreilly.com/ideas/deep-convolutional-generative-adversarial-networks-with-tensorflow
"""
import os
from glob import glob
from matplotlib import pyplot
from PIL import Image
import numpy as np

import tensorflow as tf


# download CelebA dataset with provided helper function #######################
import helper
data_dir = 'C:/Users/meyerm/Documents/OReilly/data'
helper.download_celeb_a()
# CelebA contains more than 200000 images with each 40 attribute annotations
# we ignore the annotations, because we want to generate new faces

# image configuration #########################################################
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
data_files = glob(os.path.join(data_dir, 'celebA/*.jpg'))
shape = (len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, 3)

# functions to get batches of images ##########################################
# define function to read an image
def get_image(image_path, width, height, mode):
    """
    Read an image from image_path
    Return as a numpy array
    """
    image = Image.open(image_path)
    
    if image.size != (width, height):
        # remove most pixels being not part of a face
        face_width = face_height = 108
        j = (image.size[0] - face_width) // 2  # // is floor division
        i = (image.size[1] - face_height) // 2
        image = image.crop([j, i, j + face_width, i + face_height])
        image = image.resize([width, height], Image.BILINEAR)
    
    return np.array(image.convert(mode))


# define function for batch generation
def get_batch(image_files, width, height, mode='RGB'):
    """
    Get a single image
    """
    data_batch = np.array(
                    [get_image(sample_file, width, height, mode) for sample_file in image_files]
                    ).astype(np.float32)
    
    # check if images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))
    
    return data_batch


def get_batches(batch_size):
    """
    Generate batches
    """
    IMAGE_MAX_VALUE = 255
    
    current_idx = 0
    # shape[0] is length of data_files
    while current_idx + batch_size <= shape[0]:
        data_batch = get_batch(
                        data_files[current_idx:current_idx + batch_size],
                        *shape[1:3])
        current_idx += batch_size
        
        yield data_batch / IMAGE_MAX_VALUE - 0.5

# plot test_images
test_images = get_batch(glob(os.path.join(data_dir, "celebA/*.jpg"))[:10], 56, 56)
pyplot.imshow(helper.images_square_grid(test_images))
pyplot.show()


# define network input ########################################################
def model_inputs(image_width, image_height, image_channels, z_dim):
    """
    Create the model inputs as placeholders
    """
    inputs_real = tf.placeholder(tf.float32, shape=(None, image_width, image_height, image_channels),
                        name="input_real")
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="input_z")
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    return inputs_real, inputs_z, learning_rate


# discriminator ###############################################################
def discriminator(images, reuse_variables=False):
    """
    Create the D as a convolutional network with 4 layers as in DCGAN paper
    Every convolutional layer performs:
        - convolution
        - batch normalization (for faster and more accurate training)
        - leaky ReLU activation
    Last layer:
        - dense
        - sigmoid activation
    """
    alpha = 0.2  # slope of leaky ReLU

    with tf.variable_scope('discriminator', reuse=reuse_variables):
        # conv 1
        conv1 = tf.layers.conv2d(images, 64, 5, 2, "SAME")
        lrelu1 = tf.maximum(alpha * conv1, conv1)
        
        # conv 2
        conv2 = tf.layers.conv2d(lrelu1, 128, 5, 2, "SAME")
        batch_norm2 = tf.layers.batch_normalization(conv2, training=True)
        lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)
        
        # conv 3
        conv3 = tf.layers.conv2d(lrelu2, 256, 5, 1, "SAME")
        batch_norm3 = tf.layers.batch_normalization(conv3, training=True)
        lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)
        
        # Flatten into vector
        flat = tf.reshape(lrelu3, (-1, 4*4*256))
        
        
        # Logits
        logits = tf.layers.dense(flat, 1)
        
        # Output
        out = tf.sigmoid(logits)
        
        return out, logits


# Generator ###################################################################
def generator(z, out_channel_dim, is_train=True):
    """
    Create G network, consisting of 4 layers (like DCGAN paper)
        - fully connected layer
        - 3 transposed convolutions, each with batch normalization and leaky ReLU
          activation
        - output activation is tanh
        
    Maybe replace batch normalization with virtual batch normalization
    """
    alpha = 0.2  # slope of leaky ReLU
    reuse_variables = False if is_train==True else True
    with tf.variable_scope("generator", reuse=reuse_variables):
        # first layer is fully connected
        x_1 = tf.layers.dense(z, 2*2*512)
        
        # reshape to start stack of transposed convolutions
        transposed_conv2 = tf.reshape(x_1, (-1, 2, 2, 512))
        batch_norm2 = tf.layers.batch_normalization(transposed_conv2, training=is_train)
        lrelu2 = tf.maximum(alpha * batch_norm2, batch_norm2)
        
        # transposed conv 1
        transposed_conv3 = tf.layers.conv2d_transpose(lrelu2, 256, 5, 2, padding='VALID')
        batch_norm3 = tf.layers.batch_normalization(transposed_conv3, training=is_train)
        lrelu3 = tf.maximum(alpha * batch_norm3, batch_norm3)
        
        # transposed conv 2
        transposed_conv4 = tf.layers.conv2d_transpose(lrelu3, 128, 5, 2, padding="SAME")
        batch_norm4 = tf.layers.batch_normalization(transposed_conv4, training=is_train)
        lrelu4 = tf.maximum(alpha * batch_norm4, batch_norm4)
        
        # output layer
        logits = tf.layers.conv2d_transpose(lrelu4, out_channel_dim, 5, 2, padding="SAME")
        
        out = tf.tanh(logits)
        
        return out


# Loss functions ##############################################################
def model_loss(input_real, input_z, out_channel_dim):
    """
    two separate loss functions: one for D, one for G
    D loss has two parts: first using fake images, second using real images, 
    sum of both is overall D loss
    """
    # acc to Salimans, arXiv:1606.03498: only smooth positive labels
    label_smoothing = 0.9
    
    g_model = generator(input_z, out_channel_dim)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse_variables=True)
    
    d_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                    labels=tf.ones_like(d_model_real)*label_smoothing))
    d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                    labels=tf.zeros_like(d_model_fake)))
    d_loss = d_loss_real + d_loss_fake
    
    g_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                    labels=tf.ones_like(d_model_fake)*label_smoothing))
    
    return d_loss, g_loss


# Optimization ################################################################
def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    """
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if "discriminator" in var.name]
    g_vars = [var for var in t_vars if "generator" in var.name]
    
    # optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_trainer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
                                                        d_loss, var_list=d_vars)
        g_trainer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(
                                                        g_loss, var_list=g_vars)
    
    return d_trainer, g_trainer


# Visualization ###############################################################
def show_generator_output(sess, n_images, input_z, out_channel_dim):
    """
    Show example output for the generator
    """
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])
    
    samples = sess.run(
                generator(input_z, out_channel_dim, False),
                feed_dict={input_z: example_z})
    pyplot.imshow(helper.images_square_grid(samples))
    pyplot.show()


# Training ####################################################################
def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape):
    """
    Train the GAN
    """
    input_real, input_z, _ = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    
    steps = 0
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_idx in range(epoch_count):
            for batch_images in get_batches(batch_size):
                # values range from -0.5 to 0.5 ==> rescale to -1, 1
                batch_images = batch_images * 2
                steps += 1
                print("Increased step to ", steps)
                
                # noise vector is input for generator
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                
                _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                
                if steps % 400 == 0:
                    # at the end of every 10 epochs, get and print losses
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = g_loss.eval({input_z: batch_z})
                    print("Epoch{}/{}...".format(epoch_idx+1, epochs),
                            "Discriminator Loss: {:.4f}...".format(train_loss_d),
                            "Generator Loss: {:.4f}".format(train_loss_g))
                    
                    _ = show_generator_output(sess, 9, input_z, data_shape[3])


# Hyperparameters #############################################################
batch_size = 16
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5
epochs = 2

with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, get_batches, shape)