from architecture import *
from utils import *
import scipy.io as scio
import torch
import os
import numpy as np
from option import opt

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')

# Intialize mask
mask3d_batch, input_mask = init_mask(opt.mask_path, opt.input_mask, 10)

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

def test(model):    
    test_data = LoadTest(opt.test_path)
    test_gt = test_data.cuda().float()
    model.eval()
    with torch.no_grad():
        model_out, mask1, mask2 = model(test_gt)
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    mask1 = np.transpose(mask1.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    mask2 = np.transpose(mask2.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    return pred, mask1, mask2

def main():
    model = model_generator(opt.method, opt.pretrained_model_path).cuda()
    pred, mask1, mask2 = test(model)
    name = opt.outf + 'Test_result.mat'
    name1 = opt.outf + 'mask1.mat'
    name2 = opt.outf + 'mask2.mat'

    print(f'Save reconstructed HSIs as {name}.')
    scio.savemat(name, {'pred': pred})
    scio.savemat(name1, {'mask': mask1})
    scio.savemat(name2, {'mask': mask2})

if __name__ == '__main__':
    main()