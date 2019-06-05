"""
Builds the CIFAR10 conv network

Summary of available functions:
    # compute input images and labels for training. If you would like to run
    # evaluations, use inputs() instead
    inputs, labels = distorted_inputs()

    # compute inference on the model inputs to make a prediction
    predictions = inference(inputs)

    # compute the total loss of the prediction with respect to the labels
    loss = loss(predictions, labels)

    # create a graph to run one step of training with respect to the loss
    train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input


FLAGS = tf.app.flags.FLAGS

# model parameters ############################################################
tf.app.flags.DEFINE_integer("batch_size", 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string("data_dir", "/dl/volatile/students/data/mam/CIFAR10",
                            """ Path to the CIFAR-10 data directory""")
tf.app.flags.DEFINE_boolean("use_fp16", False, """Train the model using fp16""")


# global constants describing the CIFAR10 dataset #############################
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# traing hyperparameters ######################################################
MOVING_AVERAGE_DECAY = 0.9999  # decay to use for the moving average
NUM_EPOCHS_PER_DECAY = 350.0  # epochs after which learning rate decays
LEARNING_RATE_DECAY_FACTOR = 0.1  # learning rate decay factor
INITIAL_LEARNING_RATE = 0.1  # initial learning rate


# if model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note: prefix is removed from the names
# of the summaries when visualizing a model.
TOWER_NAME = "tower"

DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"


def _activation_summary(x):
    """
    Helper function to create summaries for activations.

    Creates a summary providing a histogram of activations and measuring the
    sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    # if using multi-GPU Training, remove Tower_[0-9] from name to maintain
    # clarity of tensorboard presentation
    tensor_name = re.sub("%s_[0-9]*/" % TOWER_NAME, "", x.op.name)
    tf.summary.histogram(tensor_name + "/activations", x)
    tf.summary.scalar(tensor_name + "/sparsity", tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        tf.Variable
    """
    with tf.device("/cpu:0"):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    Helper to create an initialized Variable with weight decay.
    Note: Variable is initialized with a truncated normal distribution.
    Weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of truncated Gaussian
        wd: add L2loss weight decay multiplied by this float. If None, weight
            decay is not added for Variable.

    Returns:
        tf.Variable
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection("losses", weight_decay)

    return var


def distorted_inputs():
    """
    Construct distorted input for CIFAR training using the Reader operations

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError("Please supply a data_dir")
    data_dir = os.path.join(FLAGS.data_dir, "cifar-10-batches-bin")
    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                        batch_size=FLAGS.batch_size)

    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)

    return images, labels


def inputs(eval_data):
    """
    Construct input for CIFAR evaluation using Reader operations.

    Args:
        eval_data: bool, indicating if one should use train or eval data set

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    Raises:
        ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError("Please supply a data_dir")
    data_dir = os.path.join(FLAGS.data_dir, "cifar-10-batches-bin")
    images, labels = cifar10_input.inputs(eval_data=eval_data,
                        data_dir=data_dir,
                        batch_size=FLAGS.batch_size)

    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)

    return images, labels


def inference(images):
    """
    Build the CIFAR10 model.

    Args:
        images: Images returned from distorted_inputs() or inputs()

    Returns:
        Logits
    """
    # Note: use tf.get_variable() to share variables across multiple GPU
    # training runs, if we only use single GPU, we could replace all instances
    # of tf.get_variable() with tf.Variable()

    # conv 1
    with tf.variable_scope("conv1") as scope:
        kernel = _variable_with_weight_decay("weights",
                    shape=[5, 5, 3, 64],
                    stddev=5e-2,
                    wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _variable_on_cpu("biases", [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool 1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                padding="SAME", name="pool1")

    # norm 1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=(0.001 / 9.0), beta=0.75,
                name="norm1")


    # conv 2
    with tf.variable_scope("conv2") as scope:
        kernel = _variable_with_weight_decay("weights",
                    shape=[5, 5, 64, 64],
                    stddev=5e-2,
                    wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding="SAME")
        biases = _variable_on_cpu("biases", [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm 2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=(0.001 / 9.0), beta=0.75,
                name="norm2")

    # pool 2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                padding="SAME", name="pool2")


    # local 3
    with tf.variable_scope("local3") as scope:
        # move everything into depth to perform a single matrix multiply
        reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay("weights", shape=[dim, 384],
                        stddev=0.04, wd=0.004)
        biases = _variable_on_cpu("biases", [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)


    # local 4
    with tf.variable_scope("local4") as scope:
        weights = _variable_with_weight_decay("weights", shape=[384, 192],
                        stddev=0.04, wd=0.004)
        biases = _variable_on_cpu("biases", [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)


    # linear layer(WX + b)
    # Note: dont apply softmax here, because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts unscaled logits
    # and performs softmax internally for efficiency
    with tf.variable_scope("softmax_linear") as scope:
        weights = _variable_with_weight_decay("weights", [192, NUM_CLASSES],
                        stddev=(1/192.0), wd=None)
        biases = _variable_on_cpu("biases", [NUM_CLASSES],
                        tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(logits)

    return logits


def loss(logits, labels):
    """
    Compute the total loss

    Compute softmax-loss
    Add L2Loss, from _variable_with_weight_decay(), to all trainable variables

    Add summary for "Loss" and "loss/avg"

    Args:
        logits: logits from inference()
        labels: Labels from distorted_inputs() or inputs(), 1D tensot of shape
                [batch_size]

    Returns:
        Loss tensor of type float
    """
    # calculate the average cross entropy loss across the batch
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name="cross_entropy_per_example"
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    tf.add_to_collection("losses", cross_entropy_mean)

    # total loss is defined as cross entropy loss + all the weight decay terms
    # (L2 loss)
    # weight decay losses are defined in _variable_with_weight_decay() and get
    # added to the tf.collection "losses"
    # in tf.add_n the individual losses get summed up element-wise
    return tf.add_n(tf.get_collection("losses"), name="total_loss")


def _add_loss_summaries(total_loss):
    """
    Add summaries for losses in CIFAR10 model

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss()
    Returns:
        loss_averages_op: operation for generating moving averages of losses
    """
    # compute moving average of all individual losses and total loss
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
    losses = tf.get_collection("losses")
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # attach a scalar summary to all individual losses and total loss,
    # do the same for averaged versions of losses
    for loss in losses + [total_loss]:
        # name each loss as "(raw)" and name moving average as original loss name
        tf.summary.scalar(loss.op.name + " (raw)", loss)
        tf.summary.scalar(loss.op.name, loss_averages.average(loss))

    return loss_averages_op


def train(total_loss, global_step):
    """
    Train CIFAR10 model.

    Create an optmizer and apply to all trainable variables. Add moving average
    for all trainable variables.

    Args:
        total_loss: Total loss from loss()
        global_step: Int Variable counting the number of processed training steps
    Returns:
        train_op: operation for Training
    """
    # variables affecting learning rate
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # decay the learning rate exponentially based on number of steps
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True
    )
    tf.summary.scalar("learning_rate", lr)

    # generate moving averages of all losses and associated summaries
    loss_averages_op = _add_loss_summaries(total_loss)

    # compute gradients
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # add histograms for trainable variables
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # add histograms for gradients
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradients", grad)

    # track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    return variables_averages_op


def maybe_download_and_extract():
    """
    Download and extract the tarball
    """
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write("\r>> Downloading %s %.1f%% " % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print("Successfully downloaded", filename, statinfo.st_size, "bytes.")
    extracted_dir_path = os.path.join(dest_directory, "cifar-10-batches-bin")
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, "r:gz").extractall(dest_directory)
