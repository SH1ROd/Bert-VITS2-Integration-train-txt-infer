# coding=gbk
import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count

import soundfile
from scipy.io import wavfile
from tqdm import tqdm

global speaker_annos
speaker_annos = []

def process(item):  
    spkdir, wav_name, args = item
    speaker = spkdir.replace("\\", "/").split("/")[-1]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    if os.path.exists(wav_path) and '.wav' in wav_path:
        os.makedirs(os.path.join(args.out_dir, speaker), exist_ok=True)
        wav, sr = librosa.load(wav_path, sr=args.sr)
        soundfile.write(
            os.path.join(args.out_dir, speaker, wav_name),
            wav,
            sr
        )

def process_text(item):
    spkdir, wav_name, args = item
    speaker = spkdir.replace("\\", "/").split("/")[-1]
    wav_path = os.path.join(args.in_dir, speaker, wav_name)
    global speaker_annos
    tr_name = wav_name.replace('.wav', '')
    with open(args.out_dir+'/'+speaker+'/'+tr_name+'.lab', "r", encoding="utf-8") as file:
             text = file.read()
    text = text.replace("{NICKNAME}",'������')
    text = text.replace("{M#��}{F#��}",'��')
    text = text.replace("{M#��}{F#��}",'��')
    substring = "{M#����}{F#���}"  
    if substring in text:
        if tr_name.endswith("a"):
           text = text.replace("{M#����}{F#���}",'����')
        if tr_name.endswith("b"):
           text = text.replace("{M#����}{F#���}",'���')
    text = text.replace("#",'')   
    text = "ZH|" + text + "\n" #
    speaker_annos.append(args.out_dir+'/'+speaker+'/'+wav_name+ "|" + speaker + "|" + text)



if __name__ == "__main__":
    parent_dir = "./genshin_dataset/"
    speaker_names = list(os.walk(parent_dir))[0][1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr", type=int, default=44100, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="./genshin_dataset", help="path to source dir")
    parser.add_argument("--out_dir", type=str, default="./genshin_dataset", help="path to target dir")
    args = parser.parse_args()
    # processs = 8
    processs = cpu_count()-2 if cpu_count() >4 else 1
    pool = Pool(processes=processs)

    for speaker in os.listdir(args.in_dir):
        spk_dir = os.path.join(args.in_dir, speaker)
        if os.path.isdir(spk_dir):
            print(spk_dir)
            for _ in tqdm(pool.imap_unordered(process, [(spk_dir, i, args) for i in os.listdir(spk_dir) if i.endswith("wav")])):
                pass
            for i in os.listdir(spk_dir):
               if i.endswith("wav"):
                  pro=(spk_dir, i, args)
                  process_text(pro)
    if len(speaker_annos) == 0:
        print("transcribe error!!!")
    with open("./filelists/short_character_anno.list", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)
    print("transcript file finished.")
