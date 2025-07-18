import logging
import os
from pathlib import Path
import torch
from contextlib import nullcontext
import wandb
try:
    wandb.init(mode="disabled")
except:
    pass

from transformers import AutoConfig, AutoImageProcessor, HfArgumentParser, set_seed
from time import time

from .arguments import ModelArguments, DataArguments, RetrieverTrainingArguments as TrainingArguments
from .data import TrainDatasetForVisualEmbedding, VisualEmbedCollator
from modeling.LexSemBridgeVisualModeling import LexSemBridgeVisual
from .trainer import BiTrainer

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    # Determine model type based on name for proper initialization
    model_name = model_args.model_name_or_path.lower()
    model_type = 'other'
    if 'vit' in model_name:
        model_type = 'vit'
    elif 'clip' in model_name:
        model_type = 'clip'
    elif 'beit' in model_name:
        model_type = 'beit'
    
    # Load image processor
    try:
        image_processor = AutoImageProcessor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        logger.info(f"Loaded image processor from {model_args.model_name_or_path}")
    except Exception as e:
        logger.warning(f"Failed to load image processor: {e}")
        logger.info("Trying to load a generic image processor based on model type...")
        
        # Fallback to generic image processors based on model type
        if model_type == 'vit':
            image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        elif model_type == 'clip':
            image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        elif model_type == 'beit':
            image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
        else:
            raise ValueError(f"Could not determine appropriate image processor for {model_args.model_name_or_path}")
            
    # Load config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )
    logger.info('Config: %s', config)

    # Initialize the visual model
    model = LexSemBridgeVisual(
        model_name=model_args.model_name_or_path,
        computation_method=training_args.computation_method,
        vocabulary_filter=training_args.vocabulary_filter,
        vocab_weight_fusion_q=training_args.vocab_weight_fusion_q,
        vocab_weight_fusion_p=training_args.vocab_weight_fusion_p,
        scale=training_args.scale,
        normalized=training_args.normlized,
        image_pooling_method=training_args.sentence_pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        use_inbatch_neg=training_args.use_inbatch_neg,
        patch_num=training_args.patch_num,
        vocab_size=training_args.vocab_size
    )

    # Freeze position embeddings if requested
    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    # Create dataset and collator
    train_dataset = TrainDatasetForVisualEmbedding(args=data_args, image_processor=image_processor)
    
    collator = VisualEmbedCollator(
        image_processor=image_processor,
        query_max_len=data_args.query_max_len,
        passage_max_len=data_args.passage_max_len
    )

    # Initialize trainer
    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=image_processor  # Use image_processor as tokenizer for the trainer
    )

    # Ensure output directory exists
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    start_time = time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    trainer.train()
    
    end_time = time()
    
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Maximum GPU memory allocated: {max_memory:.2f} GB")
    
    print(f"Training time: {end_time - start_time}")

    # Save the final model
    trainer.save_model()

    # Save the image processor
    if trainer.is_world_process_zero():
        image_processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()