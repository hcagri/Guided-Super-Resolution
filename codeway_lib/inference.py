from .model import Pix2Pix
from .utils import *

import torch 
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
import os.path as osp
from PIL import Image


def inference(guide, source, _config, _run) -> Image:
    '''
    
    Args:
        guide   , tensor     : High-Resolution guide image G : [Channel,Hight,Width]
        source  , tensor     : Low-Resolution source map   S : [1, Hight, Width] 
        _config , dictionary : Configration dictionary
        _run    , run object : Sacred run object
    Return:
        target_output     : High-Resolution source map    : [Hight, Width]  
    '''

    num_channel, M, N = guide.size(0), source.size(-1), guide.size(-1)
    D = N//M

    guide = guide.to(_config['device'])
    source = source.to(_config['device'])
    pos_image = create_grid(N).to(_config['device'])

    _, _, guide = normalize(guide)
    mean_s, std_s, source = normalize(source)

    guide, pos_image = guide.unsqueeze(0), pos_image.unsqueeze(0)

    kernel_size = D
    kernel_stride = D

    guide_blocks = guide.unfold(2, kernel_size, kernel_stride).unfold(3, kernel_size, kernel_stride)
    guide_blocks = guide_blocks.contiguous().view(guide_blocks.size(1), -1, guide_blocks.size(4),guide_blocks.size(5)).permute(1,0,2,3)

    pos_image_blocks = pos_image.unfold(2, kernel_size, kernel_stride).unfold(3, kernel_size, kernel_stride)
    pos_image_blocks = pos_image_blocks.contiguous().view(pos_image_blocks.size(1), -1, pos_image_blocks.size(4),pos_image_blocks.size(5)).permute(1,0,2,3)

    source_blocks = source.view(-1,1)

    dataset = TensorDataset(torch.concat([guide_blocks, pos_image_blocks], dim=1), source_blocks)
    data_loader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=True)

    ''' Model, loss and optimizer initializations  '''

    model = Pix2Pix(in_channel=num_channel).to(_config['device'])

    optimizer = torch.optim.Adam([
                        {'params': model.upsample_pix.parameters(), 'weight_decay': 1e-3},
                        {'params': model.upsample_pos.parameters(), 'weight_decay': 1e-4},
                        {'params': model.downsample_comb.parameters(), 'weight_decay': 1e-4}
                    ], lr=1e-3)

    loss_fn = torch.nn.L1Loss()

    ''' Training Loop  '''
    iterator = tqdm(range(1,_config['num_epoch']), leave=True)
    for epoch in iterator:
        iterator.set_description_str(f"Epoch: {epoch}")
        model.train()
        loss_epoch = 0
        step = 1
        for (batch, source_batch) in data_loader:

            guide_batch, pos_image_batch = torch.split(batch, [num_channel,2], dim=1)

            optimizer.zero_grad()
            output = model(guide_batch, pos_image_batch)
            output = torch.mean(output, dim=[2,3])

            loss = loss_fn(output, source_batch)

            loss.backward()
            optimizer.step()    

            loss_epoch += loss.item()
            step += 1 
        
        iterator.set_postfix_str(f"Loss: {loss_epoch/step:.4f}")

        if epoch % _config['save_result_freq'] == 0:
            model.eval()
            with torch.no_grad():
                output = model(guide, pos_image)
                target_output = unnormalize(mean_s, std_s, output).squeeze()
                save_image(target_output, save_path=osp.join(_run.experiment_info['base_dir'], 'runs', _run._id, "results", f"epoch_{epoch}.png"))
    model.eval()
    with torch.no_grad():
        output = model(guide, pos_image)
        target_output = unnormalize(mean_s, std_s, output).squeeze()
        save_image(target_output, save_path=osp.join(_run.experiment_info['base_dir'], 'runs', _run._id, "output.png"))

    return target_output
