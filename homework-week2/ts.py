import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# hello = tf.constant("Hello, Tensorflow!")
# sess = tf.compat.v1.Session()
# print(sess.run(hello))
gpus = tf.config.experimental.list_physical_devices(device_type='CPU')

if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='CPU')