# LexSemBridgeVisualModeling.py
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel, AutoModelForMaskedImageModeling, BeitForMaskedImageModeling, ViTModel, CLIPVisionModel
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class LexSemBridgeVisual(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 computation_method: str = 'clr',
                 vocabulary_filter: bool = False,
                 vocab_weight_fusion_q: bool = True,
                 vocab_weight_fusion_p: bool = False,
                 scale: float = 1.0,
                 normalized: bool = False,
                 image_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True,
                 patch_num: int = 196,  # Default patch number (14x14 for ViT-B/16)
                 vocab_size: int = 8192  # Default vocabulary size for ViT, CLIP
                 ):
        super().__init__()
        self.computation_method = computation_method
        self.model_name = model_name
        
        # Determine model type based on model_name
        self.model_type = self._get_model_type(model_name)
        
        # Initialize the appropriate model based on model type
        if self.computation_method == 'clr':
            if self.model_type == 'beit':
                print("Using BEiT with CLR Representation Enhancement")
                self.model = BeitForMaskedImageModeling.from_pretrained(model_name, output_hidden_states=True)
            else:
                raise ValueError(f"CLR computation method requires a model with masked image modeling capability. {self.model_type} does not support this.")
        else:
            if self.model_type == 'vit':
                self.model = ViTModel.from_pretrained(model_name)
            elif self.model_type == 'clip':
                self.model = CLIPVisionModel.from_pretrained(model_name)
            elif self.model_type == 'beit':
                self.model = BeitForMaskedImageModeling.from_pretrained(model_name, output_hidden_states=True)
            else:
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        if self.computation_method == 'llr':
            self.sparse_linear = nn.Linear(in_features=self.model.config.hidden_size, out_features=1)
        else:
            self.sparse_linear = None
            
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 
            "npu" if hasattr(torch, 'npu') and torch.npu.is_available() else 
            "cpu"
        )
        
        # Set vocabulary size based on model type
        if self.model_type == 'beit':
            self.vocab_size = self.model.config.vocab_size
        else:
            self.vocab_size = vocab_size
            
        self.patch_num = patch_num

        if self.computation_method == 'slr':
            self.scale = nn.Parameter(torch.tensor(scale))

        self.vf = vocabulary_filter
        self.fusion_q = vocab_weight_fusion_q
        self.fusion_p = vocab_weight_fusion_p

        self.linearlayer = nn.Linear(self.vocab_size, self.model.config.hidden_size)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normalized = normalized
        self.image_pooling_method = image_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config

        if not normalized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")
        if normalized:
            if self.temperature > 0.5:
                raise ValueError("Temperature should be smaller than 1.0 when use cosine similarity (i.e., normalized=True). Recommend to set it 0.01-0.1")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def _get_model_type(self, model_name):
        """Determine the model type based on model name"""
        model_name_lower = model_name.lower()
        if 'vit' in model_name_lower:
            return 'vit'
        elif 'clip' in model_name_lower:
            return 'clip'
        elif 'beit' in model_name_lower:
            return 'beit'
        else:
            return 'other'

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def image_embedding(self, hidden_state, mask):
        if self.image_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.image_pooling_method == 'cls':
            return hidden_state[:, 0]
        
    def vocabulary_filter(self, embed, patch_indices, features):
        # For visual models, we use patch indices instead of input_ids
        # Here patch_indices represents which patches are active in the image
        patch_weights = torch.ones_like(patch_indices, dtype=torch.float, device=self.device) * features["attention_mask"]
        size = torch.tensor((patch_indices.size(0), self.vocab_size), device=patch_indices.device)
        patch_dense_trans = torch.zeros(size.tolist(), device=patch_indices.device, dtype=patch_weights.dtype).scatter_reduce_(1, patch_indices, patch_weights, reduce="amax")
        sorted_embed, _ = torch.sort(embed, descending=True)
        token_indices = torch.nonzero(patch_dense_trans == 1.0, as_tuple=True)[1]
        with torch.no_grad():
            embed[0, token_indices] = sorted_embed[0, 0]
        return embed
        
    def slr_enhancement(self, features):
        # For visual models, we consider patches as tokens
        # We generate patch indices based on the attention mask
        batch_size = features["pixel_values"].size(0)
        attention_mask = features.get("attention_mask", torch.ones(batch_size, self.patch_num, device=self.device))
        
        # Create sequential patch indices (similar to input_ids in text models)
        patch_indices = torch.arange(self.patch_num, device=self.device).expand(batch_size, -1)
        
        # Apply scaling to patch weights
        patch_weights = attention_mask * self.scale
        
        # Create a dense representation for patches
        size = torch.tensor((batch_size, self.vocab_size), device=self.device)
        patch_dense_trans = torch.zeros(size.tolist(), device=self.device, dtype=patch_weights.dtype).scatter_reduce_(1, patch_indices, patch_weights, reduce="amax")
        
        if self.vf:
            patch_dense_trans = self.vocabulary_filter(patch_dense_trans, patch_indices, features)
            
        sparse_des_align_vecs = torch.softmax(self.linearlayer(patch_dense_trans), dim=-1)
        
        # Get image embeddings from the model
        if self.model_type == 'beit':
            if isinstance(self.model, BeitForMaskedImageModeling):
                last_hidden_state = self.model(pixel_values=features["pixel_values"], return_dict=True).hidden_states[-1]
            else:
                last_hidden_state = self.model(pixel_values=features["pixel_values"], return_dict=True).last_hidden_state
        else:
            last_hidden_state = self.model(pixel_values=features["pixel_values"], return_dict=True).last_hidden_state
            
        p_reps = self.image_embedding(last_hidden_state, attention_mask)
        
        return sparse_des_align_vecs, p_reps

    def llr_enhancement(self, features):
        # For visual models, we consider patches as tokens
        batch_size = features["pixel_values"].size(0)
        attention_mask = features.get("attention_mask", torch.ones(batch_size, self.patch_num, device=self.device))
        
        # Create sequential patch indices (similar to input_ids in text models)
        # actual_patch_num = last_hidden_state.size(1) # to include the [CLS] token
        # patch_indices = torch.arange(actual_patch_num, device=self.device).expand(batch_size, -1)
        # patch_indices = torch.arange(self.patch_num, device=self.device).expand(batch_size, -1)
        
        # Get image embeddings from the model
        if self.model_type == 'beit':
            if isinstance(self.model, BeitForMaskedImageModeling):
                last_hidden_state = self.model(pixel_values=features["pixel_values"], return_dict=True).hidden_states[-1]
            else:
                last_hidden_state = self.model(pixel_values=features["pixel_values"], return_dict=True).last_hidden_state
        else:
            last_hidden_state = self.model(pixel_values=features["pixel_values"], return_dict=True).last_hidden_state
        
        # Create sequential patch indices
        actual_patch_num = last_hidden_state.size(1) # to include the [CLS] token
        patch_indices = torch.arange(actual_patch_num, device=self.device).expand(batch_size, -1)
        # Apply the sparse linear layer to get patch weights
        token_weights = torch.relu(self.sparse_linear(last_hidden_state))
        
        # Create a sparse embedding for patches
        sparse_embedding = torch.zeros(batch_size, actual_patch_num, self.vocab_size,
                                      dtype=token_weights.dtype,
                                      device=token_weights.device)
        sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=patch_indices.unsqueeze(-1), src=token_weights)
        
        # Take the maximum value across patches for each vocabulary token
        sparse_embedding = torch.max(sparse_embedding, dim=1).values
        
        if self.vf:
            sparse_embedding = self.vocabulary_filter(sparse_embedding, patch_indices, features)
            
        sparse_des_align_vecs = torch.softmax(self.linearlayer(sparse_embedding), dim=-1)
        p_reps = self.image_embedding(last_hidden_state, attention_mask)
        
        return sparse_des_align_vecs, p_reps

    def clr_enhancement(self, features):
        # CLR enhancement requires masked image modeling capability
        # Currently only supported for BEiT models
        if self.model_type != 'beit':
            raise ValueError("CLR enhancement only supports BEiT models with masked image modeling capability")
                
        batch_size = features["pixel_values"].size(0)
        attention_mask = features.get("attention_mask", torch.ones(batch_size, self.patch_num, device=self.device))
        
        # Create sequential patch indices
        patch_indices = torch.arange(self.patch_num, device=self.device).expand(batch_size, -1)
        
        # Get outputs from the masked image modeling model - 移除 attention_mask 参数
        model_outputs = self.model(pixel_values=features["pixel_values"], return_dict=True)
        
        # Get the last hidden state and logits for masked image prediction
        last_hidden_state = model_outputs.hidden_states[-1]
        vocabulary_weights = model_outputs.logits[:, 0, :]  # Using first token (CLS) for vocabulary prediction
        
        if self.vf:
            vocabulary_weights = self.vocabulary_filter(vocabulary_weights, patch_indices, features)
            
        vocabulary_weights = torch.log1p(torch.relu(vocabulary_weights))
        p_reps = self.image_embedding(last_hidden_state, attention_mask)
        sparse_des_align_vecs = torch.softmax(self.linearlayer(vocabulary_weights), dim=-1)
        
        return sparse_des_align_vecs, p_reps

    def encode_q(self, features):
        if features is None:
            return None
            
        if self.fusion_q:
            if self.computation_method == "slr":
                sparse_des_align_vecs, p_reps = self.slr_enhancement(features)
            elif self.computation_method == "llr":
                sparse_des_align_vecs, p_reps = self.llr_enhancement(features)
            else:  # clr
                sparse_des_align_vecs, p_reps = self.clr_enhancement(features)
            
            fusion_vector = sparse_des_align_vecs * p_reps
            if self.normalized:
                fusion_vector = torch.nn.functional.normalize(fusion_vector, dim=-1)
            return fusion_vector.contiguous()
        else:
            attention_mask = features.get("attention_mask", torch.ones(features["pixel_values"].size(0), self.patch_num, device=self.device))
            
            if self.model_type == 'beit':
                model_outputs = self.model(pixel_values=features["pixel_values"], return_dict=True)
                last_hidden_state = model_outputs.hidden_states[-1]
            else:
                model_outputs = self.model(pixel_values=features["pixel_values"], return_dict=True)
                last_hidden_state = model_outputs.last_hidden_state
                
            p_reps = self.image_embedding(last_hidden_state, attention_mask)
            if self.normalized:
                p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
            return p_reps.contiguous()

    def encode_p(self, features):
        if features is None:
            return None
            
        if self.fusion_p:
            if self.computation_method == "slr":
                sparse_des_align_vecs, p_reps = self.slr_enhancement(features)
            elif self.computation_method == "llr":
                sparse_des_align_vecs, p_reps = self.llr_enhancement(features)
            else:  # clr
                sparse_des_align_vecs, p_reps = self.clr_enhancement(features)
            
            fusion_vector = sparse_des_align_vecs * p_reps
            if self.normalized:
                fusion_vector = torch.nn.functional.normalize(fusion_vector, dim=-1)
            return fusion_vector.contiguous()
        else:
            attention_mask = features.get("attention_mask", torch.ones(features["pixel_values"].size(0), self.patch_num, device=self.device))
            
            if self.model_type == 'beit':
                model_outputs = self.model(pixel_values=features["pixel_values"], return_dict=True)
                last_hidden_state = model_outputs.hidden_states[-1]
            else:
                model_outputs = self.model(pixel_values=features["pixel_values"], return_dict=True)
                last_hidden_state = model_outputs.last_hidden_state
                
            p_reps = self.image_embedding(last_hidden_state, attention_mask)
            if self.normalized:
                p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
            return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_score: Tensor = None):
        q_reps = self.encode_q(query)
        p_reps = self.encode_p(passage)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature  # B B*G
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature  # B G

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
                 v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        
    def load(self, model_path: str):
        """Load a saved model from a given path"""
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
