# LexSemBridge

LexSemBridge: Fine-Grained Dense Representation Enhancement through Token-Aware Embedding Augmentation

[![Paper](https://img.shields.io/badge/arXiv-2508.09459-b31b1b.svg)](https://arxiv.org/abs/2508.17858)
[![License](https://img.shields.io/badge/License-apache-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()

[English](https://github.com/Jasaxion/LexSemBridge/blob/main/README.md)

## 环境准备

```bash
1. 克隆或下载整个代码仓库。
2. conda create -n lexsem python=3.10
3. conda activate lexsem
4. cd LexSemBridge
5. pip install -r requirements.txt
```

### 数据集与模型

- 数据集下载

| 训练与评估数据                                               | Huggingface文件名                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 包含 train_data, eval_data（HotpotQA, FEVER, NQ），以及 eval_visual_data（CUB200, StandfordCars） | [Jasaxion/LexSemBridge_eval](https://huggingface.co/datasets/Jasaxion/LexSemBridge_eval) |

- 下载完整数据并解压到当前目录下。
- 模型下载

⭐️当前最佳模型：

| 模型名称                   | Huggingface文件名                                            |
| -------------------------- | ------------------------------------------------------------ |
| LexSemBridge-CLR-snowflake | [Jasaxion/LexSemBridge_CLR_snowflake](https://huggingface.co/Jasaxion/LexSemBridge_CLR_snowflake) |

## 模型训练

训练参数说明：

- `nproc_per_node`：使用n个GPU运行脚本，进行分布式训练。
- `computation_method`：词表权重计算方法，可选 ['SLR', 'LLR', 'CLR']：
  - `SLR`: 统计词表示，基于token直接计算
  - `LLR`: 学习型词表示
  - `CLR`: 上下文词表示
- `scale`：词表权重缩放因子（仅SLR生效）
- `vocab_weight_fusion_q`：是否启用Query Encoder的词表权重融合
- `vocab_weight_fusion_p`：是否启用Passage Encoder的词表权重融合
- `ignore_special_tokens`：是否忽略特殊token
- `output_dir`：模型输出保存路径
- `model_name_or_path`：预训练模型名或路径
- `train_data`：训练数据路径

若使用基线模型，只需将 `vocab_weight_fusion_q` 和 `vocab_weight_fusion_p` 都设为 `False`

其他参数遵循 `transformers.HfArgumentParser`，详见：https://huggingface.co/docs/transformers/en/internal/trainer_utils#transformers.HfArgumentParser

### 用于文本稠密检索的训练示例

```bash
torchrun --nproc_per_node 8 \
    -m train.train_lexsem \
    --computation_method {slr/llr/clr} \
    --vocabulary_filter False \
    --scale 1.0 \
    --vocab_weight_fusion_q True \
    --vocab_weight_fusion_p False \
    --ignore_special_tokens True \
    --output_dir {模型输出路径} \
    --model_name_or_path {预训练模型路径} \
    --train_data ./LexSemBridge_eval/train_data/all_nli_triplet_train_data_HN.jsonl \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 64 \
    --passage_max_len 256 \
    --train_group_size 2 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_steps 5000
```

### 用于图像检索迁移的训练示例

```bash
torchrun --nproc_per_node 8 \
    -m train_visual.train_lexsemvisual \
    --computation_method {slr/llr/clr} \
    --vocabulary_filter False \
    --scale 1.0 \
    --vocab_weight_fusion_q True \
    --vocab_weight_fusion_p False \
    --output_dir {模型输出路径} \
    --model_name_or_path microsoft/beit-base-patch16-224 \
    --train_data ./LexSemBridge_eval/train_data/processed_beir_for_train/CUB_200_train/train.jsonl \
    --image_root_dir ./LexSemBridge_eval/train_data/processed_beir_for_train/CUB_200_train \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 30 \
    --per_device_train_batch_size 32 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 224 \
    --passage_max_len 224 \
    --train_group_size 2 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_steps 5000 \
    --patch_num 196 \
    --vocab_size 8192
```

## 模型评估

您可以方便地完成模型评估任务，只需根据上文 **数据集与模型** 中说明下载对应数据和模型，并运行如下脚本：

1. `cd evaluate`
2. 修改 `eval.py` 添加模型路径：

```python
  model_list = [
   # 此处添加模型名称或路径
  ]
```

1. 下载并移动 `evaluation_data` 到 `./evaluate/eval_data`
2. 执行 `python eval.py` 进行文本检索评估，执行 `python eval_visual.py` 进行图像检索评估
3. 脚本将自动对 HotpotQA、FEVER、NQ 等数据集（图像部分包括 CUB_200 和 StandfordCars）执行 Query、Keyword、Part-of-Passage 三类任务评估。结果将输出至 `evaluate/results.csv`

## 实验模型 Checkpoint

我们公开了所有实验模型的中间 checkpoint，用于复现实验结果。如果您需要下载所有 checkpoint，可以使用 openi 进行下载：

```bash
1. 安装 openi：
   pip install openi
2. 下载文件：
   openi dataset download <项目名> <文件名>
```

我们使用了8块A100完成了模型的微调训练。我们保存并发布了实验过程中的所有检查点。您可以直接下载以下模型检查点以复现实验结果。将 `<项目名>` 和 `<文件名>` 替换为下表中的内容：

| 模型Checkpoint               | 项目文件名                                          |
| ---------------------------- | --------------------------------------------------- |
| Baseline (bert)              | `My_Anonymous/LexSemBridge bert-original.zip`       |
| LexSemBridge-SLR(bert)       | `My_Anonymous/LexSemBridge bert-v4.zip`             |
| LexSemBridge-LLR(bert)       | `My_Anonymous/LexSemBridge bert-v1.zip`             |
| LexSemBridge-CLR(bert)       | `My_Anonymous/LexSemBridge bert-v7.zip`             |
| Baseline (distilbert)        | `My_Anonymous/LexSemBridge distilbert-original.zip` |
| LexSemBridge-SLR(distilbert) | `My_Anonymous/LexSemBridge distilbert-v4.zip`       |
| LexSemBridge-LLR(distilbert) | `My_Anonymous/LexSemBridge distilbert-v1.zip`       |
| LexSemBridge-CLR(distilbert) | `My_Anonymous/LexSemBridge distilbert-v7.zip`       |
| Baseline (mpnet)             | `My_Anonymous/LexSemBridge mpnet-original.zip`      |
| LexSemBridge-SLR(mpnet)      | `My_Anonymous/LexSemBridge mpnet-v4.zip`            |
| LexSemBridge-LLR(mpnet)      | `My_Anonymous/LexSemBridge mpnet-v1.zip`            |
| LexSemBridge-CLR(mpnet)      | `My_Anonymous/LexSemBridge mpnet-v7.zip`            |
| Baseline (roberta)           | `My_Anonymous/LexSemBridge roberta-original.zip`    |
| LexSemBridge-SLR(roberta)    | `My_Anonymous/LexSemBridge roberta-v4.zip`          |
| LexSemBridge-LLR(roberta)    | `My_Anonymous/LexSemBridge roberta-v1.zip`          |
| LexSemBridge-CLR(roberta)    | `My_Anonymous/LexSemBridge roberta-v7.zip`          |
| Baseline (tinybert)          | `My_Anonymous/LexSemBridge tinybert-original.zip`   |
| LexSemBridge-SLR(tinybert)   | `My_Anonymous/LexSemBridge tinybert-v4.zip`         |
| LexSemBridge-LLR(tinybert)   | `My_Anonymous/LexSemBridge tinybert-v1.zip`         |
| LexSemBridge-CLR(tinybert)   | `My_Anonymous/LexSemBridge tinybert-v7.zip`         |

## 引用

如果本项目对您有帮助，请引用：

```
@article{zhan2025lexsembridge,
  title={LexSemBridge: Fine-Grained Dense Representation Enhancement through Token-Aware Embedding Augmentation},
  author={Zhan, Shaoxiong and Lin, Hai and Tan, Hongming and Cai, Xiaodong and Zheng, Hai-Tao and Su, Xin and Shan, Zifei and Liu, Ruitong and Kim, Hong-Gee},
  journal={arXiv preprint arXiv:2508.17858},
  year={2025}
}
```
