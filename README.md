# Bert-VITS2 for windows

## 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。由使用本整合包产生的问题和作者、原作者无关！！！
 

### 请根据主目录下的训练步骤.txt中的步骤训练和推理 
### 适配windows的requirements.txt，加了个长文本分段推理和手机听书的api，
### 非本专业，屎山代码.
### 一些大文件没放进去：
### "./bert/chinese-roberta-ext-large"下的:
### "pytorch_model.bin"
### -->https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/tree/main
### "./pretrained_models"下的三个预训练模型文件:
### "D_0.pth"
### "G_0.pth"
### "DUR_0.pth"
### 以及训练好的放在"./model_saved/角色名"下的训练好的模型，
### 都可以去huggingface上获取，