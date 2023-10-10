
import flask
from flask import Flask, request

app = Flask(__name__)

import sys, os

import numpy as np

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import logging

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s")

logger = logging.getLogger(__name__)

import torch
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence, get_bert
from text.cleaner import clean_text


net_g = None


def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str)
    del word2ph

    assert bert.shape[-1] == len(phone)

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)

    return bert, phone, tone, language

def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, hps, device):
    global net_g
    bert, phones, tones, lang_ids = get_text(text, "ZH", hps)
    with torch.no_grad():
        x_tst=phones.to(device).unsqueeze(0)
        tones=tones.to(device).unsqueeze(0)
        lang_ids=lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, speakers, tones, lang_ids, bert, sdp_ratio=sdp_ratio
                           , noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        return audio

def tts_fn(text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale,hps, device, stop_time = 1.0):

    # 处理中文双引号
    text = text.replace("“", " ").replace("”", " ")
    with torch.no_grad():
        audio = infer(text, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale, sid=speaker, hps=hps, device=device)
    # 分段音频的头部添加自然停顿: stop_time=1.0 单位:秒

    blank = np.zeros((int(44100 * stop_time),), dtype=np.float64)
    audio = np.concatenate((blank, audio), axis=None)
    return "Success", hps.data.sampling_rate, audio

@app.route("/")
def hello_world():
    return "<p>Hello, BertVITS2!</p>"

from markupsafe import escape

@app.route("/<name>")
def hello(name):
    return f"Hello, {escape(name)}!"

#appi格式例子
# http://192.168.x.x:1130/role_get=kaiselin&text_get=不以物喜，不以己悲&sdp_ratio=0.2&noise_scale=0.6&noise_scale_w=0.8&length_scale=1.0&speedup_factor=1.0&pitch_factor=1.0&stop_time=1.0
@app.route("/role_get=<string:role_get>"
           "&text_get=<string:text_get>"
           "&sdp_ratio=<float:sdp_ratio>"
           "&noise_scale=<float:noise_scale>"
           "&noise_scale_w=<float:noise_scale_w>"
           "&length_scale=<float:length_scale>"
           "&speedup_factor=<float:speedup_factor>"
           "&pitch_factor=<float:pitch_factor>"
           "&stop_time=<float:stop_time>"
           )

def text2audio(role_get, text_get, sdp_ratio, noise_scale, noise_scale_w, length_scale, speedup_factor, pitch_factor, stop_time):
    global net_g
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--role", default="NaN", help="name of your role in ./model_saved")
    parser.add_argument("-m", "--model_dir", default="./model_saved/candace/G_2800.pth", help="path of your model")
    parser.add_argument("-c", "--config_dir", default="./config\config.json",help="path of your config file")
    parser.add_argument("-s", "--share", default=False, help="make link public")
    parser.add_argument("-d", "--debug", action="store_true", help="enable DEBUG-LEVEL log")

    args = parser.parse_args()
    args.role=role_get
    if args.debug:
        logger.info("Enable DEBUG-LEVEL log")
        logging.basicConfig(level=logging.DEBUG)

    if args.role != "NaN":
        config_dir = f"./model_saved/{args.role}/config.json"
        args.config_dir = config_dir
    hps = utils.get_hparams_from_file(args.config_dir)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    '''
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else (
            "mps"
            if sys.platform == "darwin" and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    '''
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    if args.role != "NaN":
        # 指定要查找的目录路径
        role_directory = f"./model_saved/{args.role}/"
        # 初始化存储文件路径的列表
        maxsteps_pth = "G_0.pth"
        # 遍历指定目录下的文件
        for root, dirs, files in os.walk(role_directory):
            for filename in files:
                if filename.endswith(".pth") and eval(filename.split('_')[1].split('.')[0]) > eval(maxsteps_pth.split('_')[1].split('.')[0]):
                    # 找到步数最大的模型
                    maxsteps_pth = filename
        model_dir = os.path.join(role_directory, maxsteps_pth)
        args.model_dir = model_dir

    _ = utils.load_checkpoint(args.model_dir, net_g, None, skip_optimizer=True)

    #解码
    from urllib.parse import unquote
    # 解码URI字符串
    text_get_de = unquote(text_get)

    convert_state, sample_rate, audio_data = tts_fn(text_get_de, role_get, sdp_ratio, noise_scale, noise_scale_w, length_scale, hps, device, stop_time)

    from scipy.io.wavfile import write
    import scipy

    # 变速变调处理 for MutiTTS 推荐参数，其他软件不需要默认两个都是1.0
    # speedup_factor = 1.6  # 两倍速
    # pitch_factor = 1.7  # 提高音调
    if speedup_factor != 1.0 and pitch_factor !=1.0:
        resampled_audio = scipy.signal.resample(audio_data, int(len(audio_data) / speedup_factor))
        adjusted_pitch_audio = np.interp(np.arange(0, len(resampled_audio), pitch_factor),np.arange(0, len(resampled_audio)), resampled_audio)
    else:
        adjusted_pitch_audio = audio_data

    # 将float64数组转换为int16类型，适用于WAV文件
    audio_data_int16 = np.int16(adjusted_pitch_audio * 32767)  # 乘以32767将范围从[-1, 1]映射到[-32767, 32767]

    # # 将音频数据保存为WAV文件
    visitor_ip = request.remote_addr

    import re

    # 定义正则表达式
    pattern = r'\d+'

    # 匹配字符串中的数字
    wave_name = re.findall(pattern, visitor_ip)

    write(f'./api-cache/{wave_name}.wav', sample_rate, audio_data_int16)

    return flask.send_file(f'./api-cache/{wave_name}.wav', mimetype='audio/wav')

if __name__ == '__main__':
    # 以ipv6启动
    # app.run(host='::', port=1130, threaded=True)
    # 以ipv4启动
    app.run(host='0.0.0.0', port=1130, threaded=True)
