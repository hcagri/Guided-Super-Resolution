from codeway_lib import *

import torch 
import torch.nn.functional as F
import os 
import os.path as osp
import numpy as np

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
SETTINGS['CAPTURE_MODE'] = 'sys'
SETTINGS.CONFIG.READ_ONLY_CONFIG=False


configs = {
    'scaling_factor': 8,
    'num_epoch' : 1000,
    'batch_size' : 32,
    'save_result_freq': 100,
    'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

dirname = osp.dirname(__file__)
experiment_dir = osp.join(dirname, 'runs')

ex = Experiment("codeway", save_git_info=False)
ex.observers.append(FileStorageObserver(experiment_dir))
ex.add_config(configs)
    

@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)
    print("data_codeway/msnet_data_depth_512_test_scaling_8.npz")
    os.makedirs(osp.join(experiment_dir, _run._id, "results"))

    # guide = torch.tensor(np.asarray(Image.open(osp.join(dirname,'data' ,'19_rgb.png'))), dtype=torch.float).permute(2,0,1)
    # target = torch.tensor(np.asarray(Image.open(osp.join(dirname,'data' ,'19_depth.png'))), dtype=torch.float).unsqueeze(0)
    data = np.load('data_codeway/msnet_data_depth_512_test_scaling_8.npz')

    patches = data['patches']
    patches_gt = data['patches_gt']

    for idx in range(patches.shape[0]):
        guide = torch.tensor(patches[idx, :,:,:], dtype=torch.float)
        target = torch.tensor(patches_gt[idx, :,:,:], dtype=torch.float)
        avg_pool = torch.nn.AvgPool2d(_config['scaling_factor'])
        source = avg_pool(target)

        save_image(guide, save_path=osp.join(experiment_dir, _run._id, "results", "guide.png"))
        save_image(target, save_path=osp.join(experiment_dir, _run._id, "results", "target.png"))
        save_image(source, save_path=osp.join(experiment_dir, _run._id, "results", "source.png"))

        target_output = inference(guide, source, _config, _run)

        target = target.squeeze()
        mae = F.l1_loss(target, target_output)
        mse = F.mse_loss(target, target_output)
        pbp = percent_bad_pixels(target_output, target, sigma=1)

        print(f"MAE: {mae.item():.4f}  |  MSE: {mse.item():.4f}  |  PBP: {pbp:.4f}")

