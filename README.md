# Bert-VITS2 for windows

## 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。由使用本整合包产生的问题和作者、原作者无关！！！
 

### 请根据主目录下的训练步骤.txt中的步骤训练和推理 
### 适配windows的requirements.txt，加了个长文本分段推理和手机听书的api，
# 本人非本专业，第一次玩，屎山代码写着玩的
# 代码主要在
# https://github.com/YYuX-1145/Bert-VITS2-Integration-package
# https://github.com/fishaudio/Bert-VITS2这个作者
# 这两个作者的基础上修改的
# 如有其他造成不便的问题或侵权问题请联系b站，本人光速删除！
### 一些大文件没放进去：
### "./bert/chinese-roberta-ext-large"下的:
### "pytorch_model.bin"
### -->https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/tree/main
### "./pretrained_models"下的三个预训练模型文件:
### "D_0.pth"
### "G_0.pth"
### "DUR_0.pth"
### -->暂无
### 以及训练好的放在"./model_saved/角色名"下的训练好的模型，
### 都可以去huggingface上获取，
### b站：https://www.bilibili.com/video/BV1nu4y1t7wA