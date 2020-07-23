# F***

import os
import fnmatch
import os.path as P

def find_recursive(root_dir, ext=['.mp4']):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[-1] in ext:
                files.append(os.path.join(root, filename))
    return files

import ffmpeg
def extract(video_path, audio_path, frame_path):
    input = ffmpeg.input(video_path, loglevel='error')
    if audio_path is not None and not P.exists(audio_path):
        audio_stream = (
            input.audio
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='11025')
            .overwrite_output()
            .run()
        )

    if frame_path is not None and not P.exists(frame_path + 'finish.txt'):
        video_stream = (
            input
            .filter('fps', fps=8)
            .output(f'{frame_path}/%d.jpg')
            .overwrite_output()
            .run()
        )
        with open(frame_path + 'finish.txt') as f:
            f.write('finish\n')
    print(video_path)

    
from tqdm import tqdm
def update(res):
    global bar
    bar.update()

if __name__ == '__main__':
    import multiprocessing as MP
    videos = find_recursive('data/solo/video', ['.webm', '.mp4', '.mkv'])
    global bar
    bar = tqdm(total=len(videos), desc='run')

    pool = MP.Pool(24)
    res = []
    for video_path in videos:
        instr = P.basename(P.dirname(video_path))
        id = P.basename(P.splitext(video_path)[0])

        audio_path = f'data/solo/audio11k/{instr}/{id}.wav'
        os.makedirs(f'data/solo/audio11k/{instr}/', exist_ok=True)
        audio_path = None
        frame_path = f'data/solo/frames8/{instr}/{id}'
        os.makedirs(frame_path, exist_ok=True)
        # frame_path = None
        res.append(pool.apply_async(extract, (video_path, audio_path, frame_path), callback=update))

    pool.close()
    pool.join()

    for r in res:
        r.get()
    print('Done')