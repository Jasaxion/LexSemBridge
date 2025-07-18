import os
import logging
import torch
from transformers.trainer import Trainer
from sentence_transformers import SentenceTransformer, models
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def save_ckpt_for_sentence_transformers(ckpt_dir, pooling_mode: str = 'cls', normlized: bool=True):
    """
    Creates a SentenceTransformer model from a checkpoint directory.
    Adapts the model for visual embeddings.
    """
    # For visual models, we need a custom adaptation
    try:
        # Create a custom module configuration for visual models
        model_config = {
            "max_seq_length": 512,
            "do_lower_case": True
        }
        
        # Use the custom module for vision models
        word_embedding_model = models.Transformer(ckpt_dir, model_args=model_config)
        
        # Configure pooling based on the specified mode
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 
                                       pooling_mode=pooling_mode)
        
        # Add normalization if required
        if normlized:
            normlize_layer = models.Normalize()
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model, normlize_layer], 
                                        device='cpu', trust_remote_code=True)
        else:
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], 
                                        device='cpu', trust_remote_code=True)
                                        
        # Save the complete model
        model.save(ckpt_dir)
        logger.info(f"Saved SentenceTransformer compatible model to {ckpt_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to save model in SentenceTransformer format: {e}")
        logger.info("The model is still saved in Hugging Face format and can be loaded directly.")


class BiTrainer(Trainer):
    """
    Trainer specialized for bi-encoder vision models.
    """
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Save the model, tokenizer, and training arguments.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        
        # Check if model has save method
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
            
        # Save image processor if available
        if hasattr(self, 'tokenizer') and self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)
            
        # Save training args
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        
        # Create SentenceTransformer compatible version
        if self.is_world_process_zero():
            try:
                save_ckpt_for_sentence_transformers(output_dir,
                                                    pooling_mode=self.args.sentence_pooling_method,
                                                    normlized=self.args.normlized)
            except Exception as e:
                logger.warning(f"Failed to save SentenceTransformer compatible model: {e}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for the bi-encoder model.
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss