import subprocess
import random
import os
from pathlib import Path
import librosa
from scipy.io import wavfile
import numpy as np
import torch
import argparse
import csv
import whisper
######################################################
role = "kesulu" # 请在这里修改说话人的名字，目前只支持中文语音

whisper_size = "medium"# 设置选用的whisper模型

raw_directory = f"./raw/{role}"# 指定要查找的目录路径

file_savepaths = f"./custom_character_voice/{role}/"
#####################################################

def resample_only_audio(filepaths, save_dir="data_dir", out_sr=44100):
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    for file_idx, filepath in enumerate(filepaths):

        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)

        print(f"Transcribing file {file_idx}: '{filepath}' to segments...")

        wav, sr = librosa.load(filepath, sr=None, offset=0, duration=None, mono=True)
        wav, _ = librosa.effects.trim(wav, top_db=40) #调整使开头不那么突然，越低切越狠，越高切越少
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = 0.98 * wav / peak
        wav2 = librosa.resample(wav, orig_sr=sr, target_sr=out_sr)
        wav2 /= max(wav2.max(), -wav2.min())

        wav_seg_name = f"{role}_{file_idx}.wav" # 修改名字
        out_fpath = save_path / wav_seg_name
        wavfile.write(out_fpath, rate=out_sr, data=(wav2 * np.iinfo(np.int16).max).astype(np.int16))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default=f"{role}")
    parser.add_argument("--whisper_size", default="medium")
    args = parser.parse_args()
    role = args.role

    import os
    # 初始化存储文件路径的列表
    file_paths = []

    # 遍历指定目录下的文件
    for root, dirs, files in os.walk(raw_directory):
        for filename in files:
            if filename.endswith(".wav"):
                # 构建文件的绝对路径并添加到列表中
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)

    resample_only_audio(file_paths, file_savepaths)