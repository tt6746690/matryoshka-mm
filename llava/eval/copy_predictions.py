import sys
import glob
import os
import shutil



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python copy_predictions.py <ckpt_dir> <upload_dir>")
    else:
        ckpt_dir = sys.argv[1]
        upload_dir = sys.argv[2]
    

    paths = glob.glob(os.path.join(ckpt_dir, 'eval/mmbench/*.xlsx'))
    if paths:
        src = paths[0]
    else:
        raise ValueError('mmbench answers for upload does not exist')
    if '/mmbench/' in src:
        task_name = 'mmbench'
    else:
        raise ValueError(f"{src} not sure what task it is.")
    l = src[src.index('results/')+8:].replace('/eval/mmbench', '').split('/')
    name = '_'.join(l)
    task_dir = os.path.join(upload_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    dst = os.path.join(task_dir, name)
    shutil.copy(src, dst)
    print(f"Copy {src}\n\t->{dst}")