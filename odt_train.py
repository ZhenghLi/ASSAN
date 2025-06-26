import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch
import torch.nn as nn
from models.assan import ASSAN
import data, utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='gloo', init_method='env://')
    model = ASSAN(sparsity=args.sp, in_chans=4, depths=[6, 6, 6, 6], embed_dim=60, width=args.img_size)
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    loss_mse = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.iters, args.min_lr)

    data_path = os.path.join(args.data_path, 'raw')
    target_path = os.path.join(args.data_path, 'gt')

    train_loader = data.build_dataset(args.dataset, data_path, target_path, batch_size=args.batch_size, num_workers=4, img_size=args.img_size, sp=args.sp)

    # Track moving average of loss values
    train_meters = {name: utils.RunningAverageMeter(0.98) for name in (["train_loss"])}

    iter = 0
    epoch = 0
    while True:
        train_loader.sampler.set_epoch(epoch)
        train_bar = utils.ProgressBar(train_loader, epoch=epoch)
        for meter in train_meters.values():
            meter.reset()

        for batch_id, (inputs, targets) in enumerate(train_bar):
            model.train()
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = loss_mse(outputs, targets)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_meters["train_loss"].update(loss.item()*512*64)
            train_bar.log(dict(**train_meters, lr=optimizer.param_groups[0]["lr"]), verbose=True)

            scheduler.step()

            iter += 1

            if iter % args.save_interval == 0:
                torch.save(model.module.state_dict(), 'odt_rec_' + str(args.sp) + '_ckpt.pth')

            if iter == args.iters:
                torch.save(model.module.state_dict(), 'odt_rec_' + str(args.sp) + '_ckpt_' + str(iter) + '.pth')
                exit(0)

        epoch += 1
        torch.save({'epoch': epoch, 'iter': iter, 'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()},
                           'odt_rec_' + str(args.sp) + '_resume.pth')


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Add data arguments
    parser.add_argument("--data-path", default="data", help="path to data directory")
    parser.add_argument("--dataset", default="ODT", help="train dataset name")
    parser.add_argument('--img-size', type=int, default=64, help='input patch size (width) of network input')
    parser.add_argument("--batch-size", default=2, type=int, help="train batch size")
    parser.add_argument("--sp", default=2, type=int, help="sparsity factor")

    # Add optimization arguments
    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument("--min-lr", default=1e-6, type=float, help="minimum learning rate")
    parser.add_argument("--iters", default=200000, type=int, help="force stop training at specified iteration")
    parser.add_argument("--valid-interval", default=200000, type=int, help="evaluate every N iterations")
    parser.add_argument("--save-interval", default=10000, type=int, help="save a checkpoint every N iterations")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
