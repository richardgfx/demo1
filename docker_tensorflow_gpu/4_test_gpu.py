import tensorflow as tf

print(tf.__version__)
print(tf.__path__)

print(tf.config.list_physical_devices('GPU'))

print(tf.test.is_built_with_cuda())

sys_details = tf.sysconfig.get_build_info()
print(sys_details)
print(sys_details["cuda_version"])
print(sys_details["cudnn_version"])

