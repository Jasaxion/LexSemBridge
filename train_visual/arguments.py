import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

@dataclass
class DataArguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=224,
        metadata={
            "help": "The maximum image size (height/width) for query images"
        },
    )

    passage_max_len: int = field(
        default=224,
        metadata={
            "help": "The maximum image size (height/width) for passage images"
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str= field(
        default=None, metadata={"help": "instruction for query (not used for image models)"}
    )
    passage_instruction_for_retrieval: str = field(
        default=None, metadata={"help": "instruction for passage (not used for image models)"}
    )
    
    image_root_dir: str = field(
        default=None, metadata={"help": "Root directory for image files"}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")
        if self.image_root_dir is None:
            # If image_root_dir is not specified, use the directory of train_data
            if os.path.isdir(self.train_data):
                self.image_root_dir = self.train_data
            else:
                self.image_root_dir = os.path.dirname(self.train_data)

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False, metadata={"help": "Freeze the parameters of position embeddings"})
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    normlized: bool = field(default=True)
    use_inbatch_neg: bool = field(default=True, metadata={"help": "use passages in the same batch as negatives"})
    scale: float = field(default=1.0, metadata={"help": "scale for the model withe the token-based method"})
    computation_method: str = field(default='None', metadata={"help": "the computation method, should be slr, llr, or clr"})
    vocabulary_filter: bool = field(default=True, metadata={"help": "filter the vocabulary weight to ensure the vocab in the input text"})
    vocab_weight_fusion_q: bool = field(default=True, metadata={"help": "weight fusion for Query Encoder"})
    vocab_weight_fusion_p: bool = field(default=False, metadata={"help": "weight fusion for Passage Encoder"})
    patch_num: int = field(default=196, metadata={"help": "number of patches in the visual model (e.g., 196 for 14x14 patches in ViT-B/16)"})
    vocab_size: int = field(default=8192, metadata={"help": "vocabulary size for visual models (default: 8192 for ViT, CLIP)"})