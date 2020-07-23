import os
import os.path as P
import glob
import argparse
import random
import fnmatch


def find_recursive(root_dir, ext=['.mp3']):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if P.splitext(filename)[-1] in ext:
                files.append(P.join(root, filename))
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_audio', default='./data/solo/audio11k',
                        help="root for extracted audio files")
    parser.add_argument('--root_frame', default='./data/solo/frames8',
                        help="root for extracted video frames")
    parser.add_argument('--root_video', default='./data/video',
                        help="root for extracted video frames")
    parser.add_argument('--fps', default=8, type=int,
                        help="fps of video frames")
    parser.add_argument('--path_output', default='./data/solo/',
                        help="path to output index files")
    parser.add_argument('--trainset_ratio', default=0.9, type=float,
                        help="80% for training, 20% for validation")
    args = parser.parse_args()

    # find all audio/frames pairs
    infos = []
    audio_files = find_recursive(args.root_audio, ext=['.mp3', '.wav'])
    for audio_path in audio_files:
        ext_name = P.splitext(audio_path)[-1]
        frame_path = audio_path.replace(args.root_audio, args.root_frame) \
                               .replace(ext_name, '')
        frame_files = glob.glob(frame_path + '/*.jpg')
        if len(frame_files) > args.fps * 20 and P.exists(frame_path + '/finish.txt'):
        # if True:
            infos.append(','.join([P.abspath(audio_path), P.abspath(frame_path), str(len(frame_files))]))
            print(frame_path)
    print('{} audio/frames pairs found.'.format(len(infos)))

    # split train/val
    n_train = int(len(infos) * 0.8)
    random.shuffle(infos)
    trainset = infos[0:n_train]
    valset = infos[n_train:]
    for name, subset in zip(['train', 'val'], [trainset, valset]):
        filename = '{}.csv'.format(P.join(args.path_output, name))
        with open(filename, 'w') as f:
            for item in subset:
                f.write(item + '\n')
        print('{} items saved to {}.'.format(len(subset), filename))

    print('Done!')

def check(root_frame, root_npy):
    import numpy as np
    data = np.load(root_npy + '/main.npy')
    print(len(glob.glob(root_frame + '/*.jpg')), data.shape)
    from PIL import Image
    # for id, path in enumerate(glob.glob(root_frame + '/*.jpg')):
    #     base = int(P.splitext(P.basename(path))[0])
    #     data0 = np.asarray(Image.open(path))
    #     data1 = data[base]#.swapcase()
    #     # print(data0.shape, data1.shape)
    #     if (data1 != data0).any():
            
    #         print(base, data1[0][0], data0[0][0])
    #         # Image.fromarray(data0).save('0.jpg')
    #         Image.fromarray(data1).save(f'{id}.jpg')
    #         if base >= 2000:
    #             break

if __name__ == '__main__':
    main()
    # x = 'accordion/0N26WnKiCIg'
    # x = 'tuba/UNFwNbFdiTI'
    # check(f'data/frames/{x}', f'data/frames_2/{x}')