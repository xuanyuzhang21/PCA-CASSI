import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--template', default='mst',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0, 1, 2, 3')

# Data specifications
parser.add_argument('--data_root', type=str, default='', help='dataset directory')

# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/mst_s/', help='saving_path')

# Model specifications
parser.add_argument('--method', type=str, default='mst_s', help='method name')
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')

# Training specifications
parser.add_argument('--batch_size', type=int, default=4, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=1e-4)

opt = parser.parse_args()
template.set_template(opt)

# dataset
opt.data_path = f"/userhome/PCA_CASSI/datasets/CAVE_1024/"
opt.mask_path = f"/userhome/PCA_CASSI/datasets/"
opt.test_path = f"/userhome/PCA_CASSI/datasets/test/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False