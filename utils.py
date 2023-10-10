import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and not skip_optimizer and checkpoint_dict['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    elif optimizer is None and not skip_optimizer:  
    #else: #Disable this line if Infer ,and enable the line upper
        new_opt_dict = optimizer.state_dict()
        new_opt_dict_params = new_opt_dict['param_groups'][0]['params']
        new_opt_dict['param_groups'] = checkpoint_dict['optimizer']['param_groups']
        new_opt_dict['param_groups'][0]['params'] = new_opt_dict_params
        optimizer.load_state_dict(new_opt_dict)
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            #assert "emb_g" not in k
            # print("load", k)
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except:
            print("error, %s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    print("load ")
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info("Saving model and optimizer state at iteration {} to {}".format(
        iteration, checkpoint_path))
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'model': state_dict,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--role', type=str, default="NaN",
                        help='name of your role')
    parser.add_argument('-c', '--config', type=str, default="./configs/config.json",
                        help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, default="./OUTPUT_MODEL",
                        help='Model name')
    parser.add_argument('--cont', dest='cont', action="store_true", default=False, help="whether to continue training on the latest checkpoint")

    args = parser.parse_args()
    if args.role == "NaN":
        model_dir = os.path.join("./logs", args.model)
    else:
        model_dir = os.path.join("./logs/", args.role)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    # print(model_dir)
    # print(config_save_path)
    # print(args.role)
    # print(os.getcwd())
    config_in_model_saved_path =''
    if args.role != "NaN":
        config_in_model_saved_path = f"./model_saved/{args.role}"
        if not os.path.exists(config_in_model_saved_path):
            os.makedirs(config_in_model_saved_path)

    if init:
        with open(config_path, "r", encoding='utf-8') as f:
            data = f.read()
        with open(config_save_path, "w", encoding='utf-8') as f:
            f.write(data)
        if args.role != "NaN":
            with open(os.path.join(config_in_model_saved_path, "config.json") , "w", encoding='utf-8') as f:
                f.write(data)
    else:
        with open(config_save_path, "r", encoding='utf-8') as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams.cont = args.cont
    return hparams


def clean_checkpoints(path_to_models='logs/OUTPUT_MODEL/', n_ckpts_to_keep=4, sort_by_time=True, role="OUTPUT_MODEL"):
    """Freeing up space by deleting saved ckpts

  Arguments:
  path_to_models    --  Path to the model directory
  n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
  sort_by_time      --  True -> chronologically delete ckpts
                        False -> lexicographically delete ckpts
  """
    path_to_models = f'logs/{role}/'
    import re
    ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
    name_key = (lambda _f: int(re.compile('._(\d+)\.pth').match(_f).group(1)))
    time_key = (lambda _f: os.path.getmtime(os.path.join(path_to_models, _f)))
    sort_key = time_key if sort_by_time else name_key
    x_sorted = lambda _x: sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith('_0.pth')],
                                 key=sort_key)
    to_del = [os.path.join(path_to_models, fn) for fn in
              (x_sorted('G')[:-n_ckpts_to_keep] + x_sorted('D')[:-n_ckpts_to_keep])]
    del_info = lambda fn: logger.info(f".. Free up space by deleting ckpt {fn}")
    del_routine = lambda x: [os.remove(x), del_info(x)]
    rs = [del_routine(fn) for fn in to_del]

def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r", encoding='utf-8') as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "r", encoding='utf-8') as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
            source_dir
        ))
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn("git hash values are different. {}(saved) != {}(current)".format(
                saved_hash[:8], cur_hash[:8]))
    else:
        open(path, "w", encoding='utf-8').write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
