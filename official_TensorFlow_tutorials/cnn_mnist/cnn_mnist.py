import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# ARCHITECTURE ################################################################
"""
Conv Layer 1: applies 32 5x5 filters with ReLU activation function
Pool Layer 1: max pooling with 2x2 filter and stride of 2 (pooling regions do
                not overlap)
Conv Layer 2: Applies 64 5x5 filters with ReLU activation function
Pool Layer 2: max pooling with 2x2 filter and stride of 2
Dense Layer 1: 1024 neurons with dropout of 0.4
Dense Layer 2: 10 neurons, one for each target class

=> can use:
    tf.layers.conv2d()
    tf.layers.max_pooling2d()
    tf.layers.dense()
"""

# CNN Model ###################################################################
def cnn_model_fn(features, labels, mode):
    """
    Model function for CNN, configures the CNN
    arguments:
        features:   takes MNIST feature data
        labels:     MNIST labels
        mode:       TRAIN, EVAL, PREDICT
    returns:
        predictions
        loss
        training operation
    """
    # Input layer #####
    # -1 for batch_size to specify dynamical computation based on input values
    # in features["x"] to treat batch_size as tunable hyper-parameter
    # e.g. in batches of 5, input_layer will contain 5*784 = 3920 values
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # conv layer 1 #####
    conv1 = tf.layers.conv2d(
        inputs=input_layer,  # must have shape [batch_size, image_height, image_width, channels]
        filters=32,  # number of filters used
        kernel_size=[5, 5],  # since both dimensions are same, one could write kernel_size=5
        padding='SAME',  # output Tensor has same height and width values as input Tensor
        activation=tf.nn.relu)
        # output tensor has shape [batch_size, 28, 28, 32]
    # pool layer 1 #####
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,  # must have shape [batch_size, image_height, image_width, channels]
        pool_size=[2, 2],
        strides=2)  # extracted sub-regions are separated by 2 pixels, for different
        # stride values for height and width, specify tuple or list, eg. stride=[3, 6]
        # output tensor has shape [batch_size, 14, 14, 32] => 2x2 filter reduces
        # height and width by 50%

    # conv layer 2 #####
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='SAME',
        activation=tf.nn.relu)
        # output tensor has shape [batch_size, 14, 14, 64]
    # pool layer 2 #####
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2)
        # output tensor has shape [batch_size, 7, 7, 64]

    # dense layer #####
    # firstly need to flatten feature map to shape [batch_size, features]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # output shape: [batch_size, 3136]
    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=1024,  # numbers of neurons in dense layer
        activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense,
        rate=0.4,  # randomly dropout 40% of the elements during training
        training=(mode == tf.estimator.ModeKeys.TRAIN))
        # output tensor has shape [batch_size, 1024]

    # logits layer #####
    # returns "raw" values for predictions => use dense layer with linear activation
    logits = tf.layers.dense(inputs=dropout, units=10)
    # output tensor has shape [batch_size, 10]

    # generate predictions for PREDICT and EVAL  mode #####
    predictions = {
        "classes": tf.argmax(input=logits, axis=1), # predicted class, digit from 0-9
        # add softmax_tensor to graph, it is used for PREDICT and
        # "logging_hook"
        "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
            # probability of being in class 0, 1, ..., 9
            # explicitly set a name to be able to set up the logging_hook later
    }
    pred_metrics_ops = {
        "train_accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"]
        )
    }
    tf.summary.scalar("train_accuracy", pred_metrics_ops["train_accuracy"][1])
    tf.summary.histogram("probabilities", predictions["probabilities"])

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # calculate loss for TRAIN and EVAL modes #####
    # for multi-class classification, often cross_entropy is used
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # another way: do a one_hot encoding of the labels and apply softmax_cross_entropy
    # onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    # loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    tf.summary.scalar("cross_entropy", loss)

    # configure training operation for TRAIN mode #####
    if mode == tf.estimator.ModeKeys.TRAIN:
        # build optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # the training operation is using the minimize method on the loss
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
            # using the global_step parameter is essential for TensorBoard Graphs
            # to work properly, it counts the number of training steps
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"])
    }
    # tf.summary.scalar("eval_accuracy", eval_metric_ops["accuracy"][1])

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                            eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # load training and eval data #####
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # returns np.array, 55000 images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  #returns np.array, 10000 images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


    # create the Estimator #####
    mnist_classifier= tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./mnist_convnet_model")


    # set up logging hook #####
    # key in following dict is of our choice
    tensors_to_log ={"probabilities" : "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)


    # Train the model #####
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,  # model will train, until specified steps are reached
        shuffle=True)  # shuffle the training data

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
    
    print("\nNow going into evaluation\n")

    # Evaluate the model and print results #####
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,  # model evaluates the metrics over one epoch of data
        shuffle=False)  # iterate through data sequentially
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
