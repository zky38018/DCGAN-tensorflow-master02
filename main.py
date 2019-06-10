
"""
main.py主要是调用前面定义好的模型、图像处理方法，来进行训练测试，程序的入口
"""import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

"""
执行main函数之前首先进行flags的解析，TensorFlow底层使用了python-gflags项目，
然后封装成tf.app.flags接口，也就是说TensorFlow通过设置flags来传递tf.app.run()所需要的参数，
我们可以直接在程序运行前初始化flags，也可以在运行程序的时候设置命令行参数来达到传参的目的。

这里主要设置了：

epoch：迭代次数
learning_rate：学习速率，默认是0.002
beta1
train_size
batch_size：每次迭代的图像数量
input_height：需要指定输入图像的高
input_width：需要指定输入图像的宽
output_height：需要指定输出图像的高
output_width：需要指定输出图像的宽
dataset：需要指定处理哪个数据集
input_fname_pattern
checkpoint_dir
sample_dir
train：True for training, False for testing
crop：True for training, False for testing
visualize


"""

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 96, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 96, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
FLAGS = flags.FLAGS

def main(_):
 # 首先是打印参数数据，然后判断输入图像的输出图像的宽是否指定，如果没有指定，则等于其图像的高。
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

#然后判断checkpoint和sample的文件是否存在，不存在则创建。
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

#然后是设置session参数。tf.ConfigProto一般用在创建session的时候，用来对session进行参数配置
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

#运行session，首先判断处理的是哪个数据集，然后对应使用不同参数的DCGAN类，这个类会在model.py文件中定义。
  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist':
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=10,
          z_dim=FLAGS.generate_test_images,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          z_dim=FLAGS.generate_test_images,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          data_dir=FLAGS.data_dir)

#show所有与训练相关的变量。
    show_all_variables()

"""判断是训练还是测试，如果是训练，则进行训练；如果不是，判断是否有训练好的model，
然后进行测试，如果没有先训练，则会提示“[!] Train a model first, then run test mode"""
    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      

    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization

    """最后进行可视化，visualize(sess, dcgan, FLAGS, OPTION)。"""
    OPTION = 1
    visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
