import mindspore
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from lenet5 import LeNet

def create_model():
  net = LeNet()
  param_dict=load_checkpoint('./pretrain.ckpt', net)
  load_param_into_net(net, param_dict)
  return net

if __name__=='__main__':
  net = create_model()
  print(net)
