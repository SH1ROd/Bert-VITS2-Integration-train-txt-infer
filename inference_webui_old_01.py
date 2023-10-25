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
import gradio as gr
import webbrowser


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

def infer(text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid):
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

def tts_fn(text_cut, text_cut_min_length, text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale):

    # 处理中文双引号
    text = text.replace("“", " ").replace("”", " ")
    #如果不是txt文件
    if not text_cut:
        with torch.no_grad():
            audio = infer(text, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale, sid=speaker)
        return "Success", (hps.data.sampling_rate, audio)
    else:

        text_segments = text.split("。")
        # 初始化存储裁切后文字的列表
        text_seg = []
        # 初始化当前段落
        current_segment = ""
        #最终合并音频
        sum_audio = np.array([],dtype='float64')
        # 遍历每个裁切后的段落，检查长度是否满足要求，并存入text_seg列表中
        for index, segment in enumerate(text_segments):
            # 如果当前段落加上这个segment的长度小于等于text_cut_min_length，则将这个segment加入当前段落
            if len(current_segment) + len(segment) + 1 <= text_cut_min_length:
                if current_segment:
                    current_segment += "。" + segment
                else:
                    current_segment = segment
            else:
                tmp = current_segment + "。"
                with torch.no_grad():
                    audio = infer(tmp, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                  length_scale=length_scale, sid=speaker)
                #分段音频的头部添加自然停顿
                blank = np.zeros((int(float(args.stop_time) * 44100),), dtype=np.float64)
                # audio = np.concatenate((blank, audio), axis=None)
                audio = np.concatenate((blank, audio), axis=None)
                sum_audio = np.concatenate((sum_audio, audio), axis=None)
                tmp = current_segment + "。\n\n"
                print(tmp)
                # if index == 0:
                #     with open("./output.txt", "w", encoding="utf-8") as f:
                #         f.write(tmp)
                # else:
                #     with open("./output.txt", "a", encoding="utf-8") as f:
                #         f.write(tmp)
                current_segment = segment
        # 将最后一个段落加入text_seg列表中
        if current_segment:
            tmp = current_segment + "。\n\n"
            # with open("./output.txt", "a", encoding="utf-8") as f:
            #     f.write(tmp)
            with torch.no_grad():
                audio = infer(tmp, sdp_ratio=sdp_ratio, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                              length_scale=length_scale, sid=speaker)
            # 分段音频的头部添加自然停顿
            blank = np.zeros((int(float(args.stop_time) * 44100),), dtype=np.float64)
            audio = np.concatenate((blank, audio), axis=None)
            sum_audio = np.concatenate((sum_audio, audio), axis=None)
        return "Success", (hps.data.sampling_rate, sum_audio)


def text_file_fn(texts_obj):
    data=''
    for file in texts_obj:
        with open(file.name, "r", encoding='utf-8') as f:
            data += '\n' + f.read()
    return gr.TextArea(value=data)


def text_cut_change_fn(flag):
    return gr.Slider(visible=flag), gr.File(visible=flag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--role", default="NaN", help="name of your role in ./model_saved")
    parser.add_argument("-m", "--model_dir", default="./model_saved/candace/G_2800.pth", help="path of your model")
    parser.add_argument("-c", "--config_dir", default="./config\config.json",help="path of your config file")
    parser.add_argument("-st", "--stop_time", default=1.0, help="stop time between sentences")
    parser.add_argument("-s", "--share", default=False, help="make link public")
    parser.add_argument("-d", "--debug", action="store_true", help="enable DEBUG-LEVEL log")

    args = parser.parse_args()
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

    speaker_ids = hps.data.spk2id
    speakers = list(speaker_ids.keys())
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                text = gr.TextArea(label="Text", placeholder="Input Text Here",
                                      value="生活就像海洋，只有意志坚强的人，才能到达彼岸。")
                speaker = gr.Dropdown(choices=speakers, value=speakers[0], label=f'Speaker with {maxsteps_pth}')
                sdp_ratio = gr.Slider(minimum=0, maximum=1, value=0.2, step=0.1, label='SDP Ratio')
                noise_scale = gr.Slider(minimum=0.1, maximum=1.5, value=0.6, step=0.1, label='Noise Scale')
                noise_scale_w = gr.Slider(minimum=0.1, maximum=1.4, value=0.8, step=0.1, label='Noise Scale W')
                length_scale = gr.Slider(minimum=0.1, maximum=2, value=1, step=0.1, label='Length Scale')
                #txt文件输入
                text_cut = gr.Checkbox(value=False, label="是否进行txt文件推理(自动裁切)")
                text_cut_min_length = gr.Slider(interactive=True, minimum=1, maximum=400, value=100, step=1, visible=False, label='每部分最长裁切长度(过大可能导致失败)')
                text_input = gr.File(interactive=True, file_count="multiple", file_types=[".txt"], visible=False, label="上传一个或多个txt文本文件(可分多次上传)")

                btn = gr.Button("Generate!", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="Message")
                audio_output = gr.Audio(label="Output Audio")

        btn.click(tts_fn,
                inputs=[text_cut, text_cut_min_length, text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale],
                outputs=[text_output, audio_output])

        text_cut.change(text_cut_change_fn,
                inputs=text_cut,
                outputs=[text_cut_min_length, text_input])

        text_input.upload(text_file_fn,
                inputs=text_input,
                outputs=text)
    
    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=args.share)
