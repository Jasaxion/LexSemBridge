import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class LexSemBridge(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 model_name: str = None,
                 computation_method: str = 'clr',
                 vocabulary_filter: bool = False,
                 vocab_weight_fusion_q: bool = True,
                 vocab_weight_fusion_p: bool = False,
                 scale: float = 1.0,
                 normalized: bool = False,
                 ignore_special_tokens: bool = False,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 use_inbatch_neg: bool = True
                 ):
        super().__init__()
        self.computation_method = computation_method
        if self.computation_method == 'clr':
            print("Using CLR Representation Enhancement")
            self.model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True, trust_remote_code=True)
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        if self.computation_method == 'llr':
            self.sparse_linear = nn.Linear(in_features=self.model.config.hidden_size, out_features=1)
        else:
            self.sparse_linear = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else 
            "npu" if hasattr(torch, 'npu') and torch.npu.is_available() else 
            "cpu"
        )
        self.vocab_size = self.model.config.vocab_size
        self.ignore_special_tokens = ignore_special_tokens

        if self.computation_method == 'slr':
            self.scale = nn.Parameter(torch.tensor(scale))

        self.vf = vocabulary_filter
        self.fusion_q = vocab_weight_fusion_q
        self.fusion_p = vocab_weight_fusion_p

        self.linearlayer = nn.Linear(self.model.config.vocab_size, self.model.config.hidden_size)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        self.normalized = normalized
        self.sentence_pooling_method = sentence_pooling_method
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
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        
    def vocabulary_filter(self, embed, input_ids, features):
        if(self.ignore_special_tokens):
            special_token_ids = torch.tensor([
                self.tokenizer.cls_token_id, 
                self.tokenizer.pad_token_id, 
                self.tokenizer.unk_token_id
            ], device=self.device)
            special_tokens_mask = torch.isin(input_ids, special_token_ids).int()
            tokenizer_weights = (torch.ones_like(input_ids, dtype=torch.float, device=self.device) * features["attention_mask"] * (1 - special_tokens_mask))
        else:
            tokenizer_weights = (torch.ones_like(input_ids, dtype=torch.float, device=self.device) * features["attention_mask"])
        size = torch.tensor((input_ids.size(0), self.vocab_size), device=input_ids.device)
        token_dense_trans = torch.zeros(size.tolist(), device=input_ids.device, dtype=tokenizer_weights.dtype).scatter_reduce_(1, input_ids, tokenizer_weights, reduce="amax")
        sorted_embed, _ = torch.sort(embed, descending=True)
        token_indices = torch.nonzero(token_dense_trans == 1.0, as_tuple=True)[1]
        with torch.no_grad():
            embed[0, token_indices] = sorted_embed[0, 0]
        return embed
        
    def slr_enhancement(self, features):
        input_ids = features["input_ids"].to(self.device) # batch_size x seq_length
        if(self.ignore_special_tokens):
            special_token_ids = torch.tensor([
                    self.tokenizer.cls_token_id, 
                    self.tokenizer.pad_token_id, 
                    self.tokenizer.unk_token_id
                ], device=self.device)
            special_tokens_mask = torch.isin(input_ids, special_token_ids).int()
            token_weights = (torch.ones_like(input_ids, dtype=torch.float, device=self.device) * features["attention_mask"] * (1 - special_tokens_mask)) * self.scale
        else:
            token_weights = (torch.ones_like(input_ids, dtype=torch.float, device=self.device) * features["attention_mask"]) * self.scale
        size = torch.tensor((input_ids.size(0), self.vocab_size), device=input_ids.device) # batch_size x vocab_size
        token_dense_trans = torch.zeros(size.tolist(), device=input_ids.device, dtype=token_weights.dtype).scatter_reduce_(1, input_ids, token_weights, reduce="amax") # batch_size x vocab_size
        if self.vf == True:
            token_dense_trans = self.vocabulary_filter(token_dense_trans, input_ids, features)
        sparse_des_align_vecs = torch.softmax(self.linearlayer(token_dense_trans),dim=-1) # batch_size x hidden_size
        last_hidden_state = self.model(**features, return_dict=True).last_hidden_state # batch_size x seq_length x hidden_size
        p_reps = self.sentence_embedding(last_hidden_state, features['attention_mask']) # batch_size x hidden_size
        return sparse_des_align_vecs, p_reps

    def llr_enhancement(self, features):
        input_ids = features["input_ids"].to(self.device) # batch_size x seq_length
        last_hidden_state = self.model(**features, return_dict=True).last_hidden_state # batch_size x seq_length x hidden_size
        token_weights = torch.relu(self.sparse_linear(last_hidden_state)) # batch_size x seq_length x 1
        sparse_embedding = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size,
                                       dtype=token_weights.dtype,
                                       device=token_weights.device) # batch_size x seq_length x vocab_size
        sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights) # batch_size x seq_length x vocab_size

        unused_tokens = [self.tokenizer.cls_token_id, self.tokenizer.pad_token_id,
                         self.tokenizer.unk_token_id]
        sparse_embedding = torch.max(sparse_embedding, dim=1).values # batch_size x vocab_size

        if self.vf == True:
            sparse_embedding = self.vocabulary_filter(sparse_embedding, input_ids, features)

        sparse_embedding[:, unused_tokens] *= 0.

        sparse_des_align_vecs = torch.softmax(self.linearlayer(sparse_embedding),dim=-1) # batch_size x hidden_size
        p_reps = self.sentence_embedding(last_hidden_state, features['attention_mask']) # batch_size x hidden_size

        return sparse_des_align_vecs, p_reps

    def clr_enhancement(self, features):
        input_ids = features["input_ids"] # batch_size x seq_length
        psg_out = self.model(**features, return_dict=True) # batch_size x seq_length x hidden_size
        last_hidden_state = psg_out.hidden_states[-1] # batch_size x seq_length x hidden_size
        vocabulary_weights = psg_out.logits[:, 0, :] # batch_size x vocab_size

        if self.vf == True:
            vocabulary_weights = self.vocabulary_filter(vocabulary_weights, input_ids, features)

        vocabulary_weights = torch.log1p(torch.relu(vocabulary_weights)) # batch_size x vocab_size
        p_reps = self.sentence_embedding(last_hidden_state, features['attention_mask']) # batch_size x hidden_size
        sparse_des_align_vecs = torch.softmax(self.linearlayer(vocabulary_weights),dim=-1) # batch_size x hidden_size
        # p_reps = self.sentence_embedding(last_hidden_state, features['attention_mask'])

        return sparse_des_align_vecs, p_reps

    def encode_q(self, features):
        if(self.fusion_q):
            if features is None:
                return None
            if(self.computation_method == "slr"):
                sparse_des_align_vecs, p_reps = self.slr_enhancement(features)
            elif(self.computation_method == "llr"):
                sparse_des_align_vecs, p_reps = self.llr_enhancement(features)
            else:
                sparse_des_align_vecs, p_reps = self.clr_enhancement(features)
            
            fusion_vector = sparse_des_align_vecs * p_reps
            if self.normalized:
                fusion_vector = torch.nn.functional.normalize(fusion_vector, dim=-1)
            return fusion_vector.contiguous()
        else:
            if features is None:
                return None
            psg_out = self.model(**features, return_dict=True)
            if(self.computation_method == "clr"):
                laset_hidden_state = psg_out.hidden_states[-1]
            else:
                laset_hidden_state = psg_out.last_hidden_state
            p_reps = self.sentence_embedding(laset_hidden_state, features['attention_mask'])
            if self.normalized:
                p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
            return p_reps.contiguous()

    def encode_p(self, features):
        if(self.fusion_p):
            if features is None:
                return None
            if(self.computation_method == "slr"):
                sparse_des_align_vecs, p_reps = self.slr_enhancement(features)
            elif(self.computation_method == "llr"):
                sparse_des_align_vecs, p_reps = self.llr_enhancement(features)
            else:
                sparse_des_align_vecs, p_reps = self.clr_enhancement(features)
            
            fusion_vector = sparse_des_align_vecs * p_reps
            if self.normalized:
                fusion_vector = torch.nn.functional.normalize(fusion_vector, dim=-1)
            return fusion_vector.contiguous()
        else:
            if features is None:
                return None
            psg_out = self.model(**features, return_dict=True)
            if(self.computation_method == "clr"):
                laset_hidden_state = psg_out.hidden_states[-1]
            else:
                laset_hidden_state = psg_out.last_hidden_state
            p_reps = self.sentence_embedding(laset_hidden_state, features['attention_mask'])
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
                scores = self.compute_similarity(q_reps, p_reps) / self.temperature # B B*G
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = self.compute_similarity(q_reps[:, None, :,], p_reps.view(q_reps.size(0), group_size, -1)).squeeze(1) / self.temperature # B G

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