
import torch
torch.set_printoptions(precision=10)
import numpy as np
from maskrcnn_benchmark.modeling.backbone.face_backbone.y2MobileFaceNet import y2
from maskrcnn_benchmark.config import face_cfg as cfg

from torch.autograd import Variable

def Pytroch2Caffe():
    import maskrcnn_benchmark.PytorchToCaffe.pytorch_to_caffe as pytorch_to_caffe
    PytorchParam = "/data2/hongwei/training_dir/test_101/BACKBONE/BACKBONE_0090000.pth"
    config_file = "configs/face_reg/face_net_msra_celeb.yaml"
    input_size =112
    name = "y2"

    cfg.merge_from_file(config_file)
    cfg.freeze()
    PytorchNet    = y2(cfg)
    PytorchTensor = Variable(torch.ones((1,3,input_size,input_size)) )
    torch.load(PytorchParam)
    PytorchNet.load_state_dict(torch.load(PytorchParam))
    PytorchNet.eval()
    # PytorchOutPut = PytorchNet(PytorchTensor)
    pytorch_to_caffe.trans_net(PytorchNet, PytorchTensor, name)
    Protxt= '{}.prototxt'.format(name)
    CafeeModel = '{}.caffemodel'.format(name)
    pytorch_to_caffe.save_prototxt(Protxt)
    pytorch_to_caffe.save_caffemodel(CafeeModel)
    output = PytorchNet(PytorchTensor)
    print(output.cpu().detach().numpy() )

def caffe_test():
    # del pytorch_to_caffe
    import caffe  # noqa
    input_size = 112
    prototxt = "/data/hongwei/face_benchmark/y2.prototxt"
    caffemodel = "/data/hongwei/face_benchmark/y2.caffemodel"
    net = caffe.Classifier(prototxt, caffemodel)
    caffe.set_mode_cpu()
    img = np.ones((3, input_size, input_size))
    input_data = net.blobs["blob1"].data[...]
    net.blobs['blob1'].data[...] = img

    prediction = net.forward()
    print(prediction)

if __name__ == "__main__":
    Pytroch2Caffe()
    # caffe_test()

# -0.6367888   -0.5164617
#-0.6368825 -0.5164