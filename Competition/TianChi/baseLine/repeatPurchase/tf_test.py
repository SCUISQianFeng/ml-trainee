
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')

assert len(physical_devices) > 0

tf.config.experimental.set_memory_growth(physical_devices[0], True)


if __name__ == "__main__":
    print(tf.test.is_gpu_available())
