# LexSemBridge

[中文](https://github.com/Jasaxion/LexSemBridge/blob/main/README_CN.md)

## Preparation

```
1. You need to clone or download the entire repository.
2. conda create -n lexsem python=3.10
3. conda activate lexsem
4. cd LexSemBridge
5. pip install -r requirements.txt
```

### Dataset and Model

- Dataset Download

| Training and Evaluation Data                                 | File Name (on huggingface)                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Includes train_data, eval_data (HotpotQA, FEVER, NQ), eval_visual_data(CUB200, StandfordCars). | [Jasaxion/LexSemBridge_eval](https://huggingface.co/datasets/Jasaxion/LexSemBridge_eval) |

- Download the complete data and then extract it to the current folder.

- Model Download

⭐️Current Best Model：

| Model Name                | File Name (on huggingface) |
| ------------------------- | ----------------------- |
| LexSemBridge-CLR-snowflake | [Jasaxion/LexSemBridge_CLR_snowflake](https://huggingface.co/Jasaxion/LexSemBridge_CLR_snowflake) |

## Model Training

Parameters:
`nproc_per_node`: Runs the script using n GPUs, utilizing distributed training.

`computation_method`: {Vocab weight computation method available: ['SLR', 'LLR', 'CLR']}: The method used for computing vocabulary weights. Options:

- `SLR`: Statistical Lexical Representation, direct token-based computation.
- `LLR`: Learned Lexical Representation
- `CLR`: Contextual Lexical Representation

`scale 1.0`: Scaling factor for vocabulary weights (if using SLR)

`vocab_weight_fusion_q True`: Enables vocabulary weight fusion for Query Encoder during training.

`vocab_weight_fusion_p False`: Disables vocabulary weight fusion for Passage Encoder.

`ignore_special_tokens True`: Whether Special Tokens should be ignored in computations.

`output_dir {model_output_dir}`: Path where the trained model and checkpoints will be saved.

`model_name_or_path {base_model_name or model_path}`: Pre-trained model or path to an existing model that will be trained.

`train_data {training data path}`: Path to the training data.

For Baseline, just set `vocab_weight_fusion_q` and `vocab_weight_fusion_p` to `False`

All other parameters follow the `transformers.HfArgumentParser`. For more details, please see: https://huggingface.co/docs/transformers/en/internal/trainer_utils#transformers.HfArgumentParser

### For Text Dense Retrieval

```bash
torchrun --nproc_per_node 8 \
    -m train.train_lexsem \
    --computation_method {Vocab weight computation method avaliable:['slr', 'llr', 'clr']} \
    --vocabulary_filter False \
    --scale 1.0 \
    --vocab_weight_fusion_q True \
    --vocab_weight_fusion_p False \
    --ignore_special_tokens True \
    --output_dir {model_output_dir} \
    --model_name_or_path {base_model_name or model_path} \
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

### For Image Retriever Migration

```bash
torchrun --nproc_per_node 8 \
    -m train_visual.train_lexsemvisual \
    --computation_method {Vocab weight computation method avaliable:['slr', 'llr', 'clr']} \
    --vocabulary_filter False \
    --scale 1.0 \
    --vocab_weight_fusion_q True \
    --vocab_weight_fusion_p False \
    --output_dir {model_output_dir} \
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


## Evaluation

You can easily complete all model evaluation tasks. You just need to download the relevant evaluation data and model checkpoints, as shown in the **Dataset and Model** section, and then use the following evaluation script to complete the LexSemBridge experiment evaluation.

1. `cd evaluate`
2. Add Model Name or Model Path in `eval.py`
```python
  model_list = [
   #Note: Add model name or Model Path Here
  ]
```
3. download and move `evaluation_data` to `./evaluate/eval_data`
4. Run `python eval.py` for text retrieval and `python eval_visual.py` for image retriever;
5. The script will then automatically complete the experiment evaluation for the Query, Keyword, and Part-of-Passage tasks on the HotpotQA, FEVER, and NQ datasets (same for image part with CUB_200 and StandfordCars). (The results will be outputted to evaluate/results.csv.)

## Experimental model checkpoint

We publicly release all model checkpoints during the experiment, you can use these models to reproduce the experimental results. If you need all the model checkpoints, we have uploaded all the checkpoints to the openi repository. You can download them by following the steps below:

```
1. First, install openi.
   pip install openi
2. Then, download the files.
   openi dataset download <Project> <File Name>
You need to replace <Project> and <File Name> according to the content in the table below.
```

We used 8 X A100 to complete the fine-tuning training of the model. We save and publish all checkpoints from the experimental process. You can directly download the following model checkpoints to reproduce the experimental results.

| Model Checkpoint                     | Project File Name                                   |
| ------------------------------------ | --------------------------------------------------- |
| Baseline (bert)                      | `My_Anonymous/LexSemBridge bert-original.zip`       |
| LexSemBridge-SLR-based(bert)         | `My_Anonymous/LexSemBridge bert-v4.zip`             |
| LexSemBridge-LLR-based(bert)         | `My_Anonymous/LexSemBridge bert-v1.zip`             |
| LexSemBridge-CLR-based(bert)         | `My_Anonymous/LexSemBridge bert-v7.zip`             |
| Baseline (distilbert)                | `My_Anonymous/LexSemBridge distilbert-original.zip` |
| LexSemBridge-Token-based(distilbert) | `My_Anonymous/LexSemBridge distilbert-v4.zip`       |
| LexSemBridge-LLR-based(distilbert)   | `My_Anonymous/LexSemBridge distilbert-v1.zip`       |
| LexSemBridge-CLR-based(distilbert)   | `My_Anonymous/LexSemBridge distilbert-v7.zip`       |
| Baseline (mpnet)                     | `My_Anonymous/LexSemBridge mpnet-original.zip`      |
| LexSemBridge-SLR-based(mpnet)        | `My_Anonymous/LexSemBridge mpnet-v4.zip`            |
| LexSemBridge-LLR-based(mpnet)        | `My_Anonymous/LexSemBridge mpnet-v1.zip`            |
| LexSemBridge-CLR-based(mpnet)        | `My_Anonymous/LexSemBridge mpnet-v7.zip`            |
| Baseline (roberta)                   | `My_Anonymous/LexSemBridge roberta-original.zip`    |
| LexSemBridge-SLR-based(roberta)      | `My_Anonymous/LexSemBridge roberta-v4.zip`          |
| LexSemBridge-LLR-based(roberta)      | `My_Anonymous/LexSemBridge roberta-v1.zip`          |
| LexSemBridge-CLR-based(roberta)      | `My_Anonymous/LexSemBridge roberta-v7.zip`          |
| Baseline (tinybert)                  | `My_Anonymous/LexSemBridge tinybert-original.zip`   |
| LexSemBridge-SLR-based(tinybert)     | `My_Anonymous/LexSemBridge tinybert-v4.zip`         |
| LexSemBridge-LLR-based(tinybert)     | `My_Anonymous/LexSemBridge tinybert-v1.zip`         |
| LexSemBridge-CLR-based(tinybert)     | `My_Anonymous/LexSemBridge tinybert-v7.zip`         |

## Citation

If this work is helpful, please kindly cite as:

```
@article{zhan2025lexsembridge,
  title={LexSemBridge: Fine-Grained Dense Representation Enhancement through Token-Aware Embedding Augmentation},
  author={Zhan, Shaoxiong and Lin, Hai and Tan, Hongming and Cai, Xiaodong and Zheng, Hai-Tao and Su, Xin and Shan, Zifei and Liu, Ruitong and Kim, Hong-Gee},
  journal={arXiv preprint arXiv:2508.17858},
  year={2025}
}
```
