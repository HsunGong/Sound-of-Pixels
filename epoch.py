import torch
import os
import time

from utils import AverageMeter, makedirs

def evaluate(netWrapper, loader, history, epoch, args):    
    loader.dataset.split = 'eval'
    netWrapper.eval()
    loss = AverageMeter()

    with torch.no_grad():
        torch.cuda.synchronize()
        for i, batch_data in enumerate(loader):
            # measure data time
            torch.cuda.synchronize()
            err, _ = netWrapper.forward(batch_data['audios'], batch_data['audio_mix'], batch_data['frames'], args)
            loss.update(err.mean().item())
            torch.cuda.synchronize()

        history['val']['err'].append(loss.average())
        history['val']['epoch'].append(epoch)
        inference(netWrapper, loader.dataset, history, epoch, args)
        print(f'Epoch: [{epoch}] Loss: {loss.average():.4f}')

# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, args):
    # switch to train mode
    netWrapper.train()
    loader.dataset.split = 'train'

    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss = AverageMeter()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        netWrapper.zero_grad()
        # forward pass

        err, _ = netWrapper.forward(batch_data['audios'], batch_data['audio_mix'], batch_data['frames'], args)
        err = err.mean()

        # backward
        err.backward()

        if args.clip_norm > 0:
            if type(netWrapper) == torch.nn.DataParallel:
                _n = netWrapper.module
            else:
                _n = netWrapper
            torch.nn.utils.clip_grad_norm_(
                _n.net_sound.parameters(), args.clip_norm)
            del _n

        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        
        batch_time.update(time.perf_counter() - tic)
        loss.update(err.item())
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0:
            print(f'Epoch: [{epoch}][{i}/{args.epoch_iters}],' 
                f'Time: {batch_time.average():.2f}, Data: {data_time.average():.2f}, '
                f'loss: {loss.average():.4f}'
            )

            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(loss.average())

def inference(netWrapper, dataseter, history, epoch, args):
    from librosa.output import write_wav
    # switch to train mode
    netWrapper.eval()
    dataseter.split = 'infer'
    with torch.no_grad():
        for i, batch_data in enumerate(dataseter):
            if i >= args.num_vis:
                break

            info = batch_data['info']
            mix = batch_data['audio_mix']
            audios = batch_data['audios']
            frames = batch_data['frames']

            # prepare
            output_dir = args.vis + '/' + info['id'].str.cat(sep='-')
            makedirs(output_dir)

            # reference
            # info.apply(lambda r: write_wav(output_dir + '/' + r['id'] + '.wav', r['audio'], args.audRate), axis=1)
            for idx, _ref in enumerate(audios):
                write_wav(output_dir + f'/{idx}ref.wav', _ref, args.audRate)
            write_wav(output_dir + '/mix.wav', mix, args.audRate)

            # infer -> squeeze batch
            mix = torch.Tensor(mix).unsqueeze(0).to(args.device)
            audios = torch.Tensor(audios).unsqueeze(0).to(args.device)
            frames = torch.Tensor(frames).unsqueeze(0).to(args.device)
            # batch_data['audios'], batch_data['audio_mix'], batch_data['frames']

            err, pred_audios = netWrapper.forward(audios, mix, frames, args) # num_mix, batch, len
            pred_audios = torch.stack(pred_audios)
            norm = torch.norm(mix, float('inf'))
            for idx, pred in enumerate(torch.squeeze(pred_audios.squeeze(1).detach().cpu())):
                #norm
                pred = pred - torch.mean(pred)
                pred = pred * norm / torch.max(torch.abs(pred))
                write_wav(output_dir+f'/{idx}test.wav', pred.numpy(), args.audRate)

    print('infer done')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        '-mix_scp', type=str, default='1_mix.wav', help='Path to mix scp file.')
    parser.add_argument(
        '-yaml', type=str, default='./config/Dual_RNN/train.yml', help='Path to yaml file.')
    parser.add_argument(
        '-model', type=str, default='./checkpoint/Dual_Path_RNN/best.pt', help="Path to model file.")
    parser.add_argument(
        '-gpuid', type=str, default='0', help='Enter GPU id number')
    parser.add_argument(
        '-save_path', type=str, default='./test', help='save result path')
    args=parser.parse_args()
    gpuid=[int(i) for i in args.gpuid.split(',')]
    separation=Separation(args.mix_scp, args.yaml, args.model, [])
    separation.inference(args.save_path)
