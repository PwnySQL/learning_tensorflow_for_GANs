""" Routine for decoding CIFAR-10 binary file format """

import os

import tensorflow as tf

# images are processed with this size
# Note: it is different from original CIFAR size of 32x32
IMAGE_SIZE = 32

# global constants describing the CIFAR10 data set
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    """
    Reads and parses examples from CIFAR10 data files
    For N-way reading parallism, call this function 10 times getting you
    N independent readers reading different files & positions within the data
    files -> better mixing of examples

    Args:
        filename_queue: A queue of strings with the filenames to read from

    Returns:
        An object representing a single example with the following fields:
            height: number of rows in the result (32)
            width: number of cols in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
                for this example
            label: an int32 Tensor with the label in range 0..9
            uint8image: a [height, width, depth] uint8 Tensor with the
                image data
    """

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record

    # dimensions of the images in the CIFAR10 dataset
    # description: http://www.cs.toronto.edu/~kriz/cifar.html
    label_bytes = 1  # 2 for CIFAR100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth

    # every record consists of a label followed by the image
    record_bytes = label_bytes + image_bytes

    # read a record, getting filenames from filename_queue
    # CIFAR10 has no header_bytes and footer_bytes => leave them at default == 0
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # convert from string to a vector of uint8 that is record_bytes long
    record_bytes = tf.decode_raw(value, tf.uint8)

    # first bytes represent label, which we convert from uint8->tf.int32
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32
    )

    # remaining bytes after label represent image, which we reshape from
    # [depth * height * width] to [depth, height, width]
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
            [label_bytes + image_bytes]),
        [result.depth, result.height, result.width]
    )
    # convert from [depth, height, width] to [height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """
    Construct a queued batch of images and labels.

    Args:
        image: 3D Tensor of [height, width, 3] of type.float32
        label: 1D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
            in the queue that provides of batches of examples
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue

    Returns:
        images: Images. 4D Tensor of [batch_size, height, width, 3] shape
        labels: Labels. 1D Tensor of [batch_size] shape
    """
    # create a queue that shuffles the examples, then read 'batch_size' many
    # images + labels from example queue
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )

    # display training images in the visualizer
    tf.summary.image("images", images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    """
    Construct distorted input for CIFAR train using the reader operations

    Args:
        data_dir: Path to the CIFAR10 directory
        batch_size: Number of images per batch

    Returns:
        images: Images. 4D Tensor of [batch_size, height, width, 3] shape
        labels: Labels. 1D Tensor of [batch_size] shape
    """
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i)
                for i in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to finde file: " + f)

    # create a queue that produces filenames to read
    filename_queue = tf.train.string_input_producer(filenames)

    with tf.name_scope("data_augmentation"):
        # read examples from files in filename_queue
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # image pre-processing for training the networks, note: many random
        # distortions applied to the image:

        # randomly crop a [height, width] section of the image
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # randomly flip the image horizontally
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # these operations are not commutative, consider randomizing the order
        # of their operation
        # NOTE: might have no effect, see tensorflow#1458
        distorted_image = tf.image.random_brightness(distorted_image,
                                                    max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                    lower=0.2, upper=1.8)

        # subtract the mean and divide by variance of the pixels
        float_image = tf.image.per_image_standardization(distorted_image)

        # set shape of tensors
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # ensure that random shuffling has good mixing properties
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                min_fraction_of_examples_in_queue)
        print("Filling queue with %d CIFAR images before starting to train. "
            "This will take a few minutes." % min_queue_examples)

    # generate a batch of images and labels by building up a queue of examples
    return _generate_image_and_label_batch(float_image, read_input.label,
                                            min_queue_examples, batch_size,
                                            shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """
    Construct input for CIFAR evaluation using the reader operations

    Args:
        eval_data: bool, indicating if one should use train or eval data set
        data_dir: path to the CIFAR10 data directory
        batch_size: Number of images per batch.

    Returns:
        images: Images. 4D Tensor of [batch_size, height, width, 3] shape
        labels: Labels. 1D Tensor of [batch_size] shape
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i)
                        for i in range(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, "test_batch.bin")]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)

    with tf.name_scope("input"):
        # create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # read examples from files in the filename queue
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # image processing for evaluation
        # crop central [height, width] of the images
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                                height, width)

        # subtract the mean and divide by the variance of the pixels
        float_image = tf.image.per_image_standardization(resized_image)

        # set the shape of tensors
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # ensure that the random shuffling has good mixing properties
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                min_fraction_of_examples_in_queue)

    # generate batch of images and labels by building queue of examples
    return _generate_image_and_label_batch(float_image, read_input.label,
                                            min_queue_examples, batch_size,
                                            shuffle=False)
