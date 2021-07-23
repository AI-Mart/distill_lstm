# Distilling Knowledge From Fine-tuned ernie-gram into Bi-LSTM

以下是本例的简要目录结构及说明：
```
.
├── student_lstm_small.py            # 小模型结构以及对小模型单独训练的脚本
├── teacher_train_glue.py            # 训练教师模型的脚本 
├── teacher_student_distill.py       # 用教师模型BERT蒸馏学生模型的蒸馏脚本
├── data.py               # 定义了dataloader等数据读取接口
├── utils.py              # 定义了将样本转成id的转换接口
├── args.py               # 参数配置脚本
└── README.md             # 文档，本文件
```

## 简介
### 第一步、下载词典
```shell
wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt
```

### 第二步、训练教师模型
```shell
export TASK_NAME=ChnSentiCorp
python -u ./teacher_train_glue.py \
    --model_type ernie-gram \
    --model_name_or_path ernie-gram-zh \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 5e-5 \
    --num_train_epochs 30 \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir ./teacher/$TASK_NAME/ \
    --device gpu # or cpu

```

### 第三步、蒸馏模型
这一步是将教师模型ernie-gram-zh的知识蒸馏到基于BiLSTM的学生模型中，可以运行下面的命令分别基于ChnSentiCorp对基于BiLSTM的学生模型进行蒸馏。

```shell
CUDA_VISIBLE_DEVICES=0
python teacher_student_distill.py \
    --task_name chnsenticorp \
    --vocab_size 1256608 \
    --max_epoch 6 \
    --lr 1.0 \
    --dropout_prob 0.1 \
    --batch_size 64 \
    --model_name ernie-gram-zh \
    --teacher_dir ./teacher/ChnSentiCorp/chnsenticorp_ft_model_9000.pdparams \
    --vocab_path senta_word_dict.txt \
    --output_dir distilled_models \
    --save_steps 10000 \
    
```

各参数的具体说明请参阅 `args.py` ，注意在训练不同任务时，需要调整对应的超参数。

## 蒸馏实验结果
本蒸馏实验基于GLUE的中文情感分类ChnSentiCorp数据集。实验效果均使用每个数据集的验证集（dev）进行评价，评价指标是准确率（acc），其中QQP中包含f1值。利用基于ernie-gram-zh的教师模型去蒸馏基于Bi-LSTM的学生模型，对比Bi-LSTM小模型单独训练，在ChnSentiCorp(中文情感分类)任务上分别有1.4%的提升。

## 参考文献

Tang R, Lu Y, Liu L, Mou L, Vechtomova O, Lin J. [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)[J]. arXiv preprint arXiv:1903.12136, 2019.
