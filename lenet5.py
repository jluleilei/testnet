import mindspore
from mindspore import Tensor, nn, context, Model
from mindspore.train.callback import LossMonitor
import mindspore.ops.operations as P
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.transforms.vision.c_transforms as CV
from mindspore.dataset.transforms.vision import Inter
from mindspore.common import dtype as mstype

def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """ create dataset for train or test
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # resize images to (32, 32)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)  # normalize images
    rescale_op = CV.Rescale(rescale, shift)  # rescale images
    hwc2chw_op = CV.HWC2CHW()  # change shape from (height, width, channel) to (channel, height, width) to fit network.
    type_cast_op = C.TypeCast(mstype.int32)  # change data type of label to int32 to fit network
    
    # apply map operations on images
    mnist_ds = mnist_ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=resize_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_nml_op, num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds

class LeNet(nn.Cell):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5, weight_init='TruncatedNormal', bias_init='TruncatedNormal', pad_mode='valid')
    self.conv2 = nn.Conv2d(6, 16, 5, weight_init='TruncatedNormal', bias_init='TruncatedNormal', pad_mode='valid')
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.reshape = P.Reshape()
    self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init='TruncatedNormal', bias_init='TruncatedNormal')
    self.fc2 = nn.Dense(120, 84, weight_init='TruncatedNormal', bias_init='TruncatedNormal')
    self.fc3 = nn.Dense(84, 10, weight_init='TruncatedNormal', bias_init='TruncatedNormal')

  def construct(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.reshape(x, (32, 400))
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    return x

if __name__ == '__main__':
  import numpy as np
  context.set_context(mode=context.GRAPH_MODE, save_graphs=True)
  dataset = create_dataset('/fzl/mnist/train')
  net = LeNet()
  data = Tensor(np.ones((32, 1, 32, 32)), mindspore.float32)
  y = net(data)
  net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
  lr = 0.01
  momentum = 0.9
  opt = nn.Momentum(net.trainable_params(), lr, momentum)
  mod = Model(net, loss_fn=net_loss, optimizer=opt)
  mod.train(10, dataset, callbacks=[LossMonitor(),], dataset_sink_mode=False)
