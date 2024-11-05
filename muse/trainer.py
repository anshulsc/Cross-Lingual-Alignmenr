import os
from logging import getLogger
import scipy
import scipy.linalg
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Tuple, Optional

from utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings, clip_parameters
from dico_build import build_translation_dictionary
from word_translation import load_identical_char_dico, load_dictionary

logger = getLogger(__name__)

class Trainer:
    def __init__(self, src_emb: nn.Embedding, tgt_emb: nn.Embedding, mapping: nn.Module, discriminator: Optional[nn.Module], params):
        self.params = params
        self.src_emb = src_emb.to(params.device)
        self.tgt_emb = tgt_emb.to(params.device)
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping = mapping.to(params.device)
        self.discriminator = discriminator.to(params.device) if discriminator is not None else None

        # Set up optimizers with weight decay
        self.map_optimizer = torch.optim.AdamW(self.mapping.parameters(), lr=params.lr, weight_decay=params.weight_decay)
        self.dis_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=params.lr, weight_decay=params.weight_decay) if self.discriminator else None

        # Set up learning rate schedulers with warmup
        total_steps = params.n_epochs * params.epoch_size
        self.map_scheduler = self.get_scheduler_with_warmup(self.map_optimizer, params.warmup_steps, total_steps)
        self.dis_scheduler = self.get_scheduler_with_warmup(self.dis_optimizer, params.warmup_steps, total_steps) if self.discriminator else None

        self.best_valid_metric = float('-inf')
        self.gradient_clip_val = params.gradient_clip_val

    def get_scheduler_with_warmup(self, optimizer, warmup_steps, total_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        
        return LambdaLR(optimizer, lr_lambda)

    def get_discriminator_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self.params.batch_size
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))

        src_ids = torch.randint(0, len(self.src_dico) if mf == 0 else mf, (bs,), device=self.params.device)
        tgt_ids = torch.randint(0, len(self.tgt_dico) if mf == 0 else mf, (bs,), device=self.params.device)

        src_emb = self.mapping(self.src_emb(src_ids))
        tgt_emb = self.tgt_emb(tgt_ids)

        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.cat([torch.ones(bs, device=self.params.device) * (1 - self.params.dis_smooth),
                       torch.ones(bs, device=self.params.device) * self.params.dis_smooth])

        return x, y

    def train_discriminator(self, stats: Dict[str, list]):
        self.discriminator.train()

        x, y = self.get_discriminator_batch()
        preds = self.discriminator(x)
        loss = nn.functional.binary_cross_entropy(preds, y)

        self.dis_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip_val)
        self.dis_optimizer.step()
        self.dis_scheduler.step()

        clip_parameters(self.discriminator, self.params.dis_clip_weights)

        stats['DIS_COSTS'] = stats.get('DIS_COSTS', []) + [loss.item()]
        stats['DIS_LR'] = stats.get('DIS_LR', []) + [self.dis_scheduler.get_last_lr()[0]]


    def train_mapping(self, stats: Dict[str, list]) -> int:
        if self.params.dis_lambda == 0:
            return 0

        self.discriminator.eval()

        x, y = self.get_discriminator_batch()
        preds = self.discriminator(x)
        loss = self.params.dis_lambda * nn.functional.binary_cross_entropy(preds, 1 - y)

        self.map_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mapping.parameters(), self.gradient_clip_val)
        self.map_optimizer.step()
        self.map_scheduler.step()

        self.orthogonalize()

        stats['MAP_COSTS'] = stats.get('MAP_COSTS', []) + [loss.item()]
        stats['MAP_LR'] = stats.get('MAP_LR', []) + [self.map_scheduler.get_last_lr()[0]]


        return 2 * self.params.batch_size

    def load_training_dictionary(self, dico_train: str):
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        elif dico_train == "default":
            filename = f'{self.params.src_lang}-{self.params.tgt_lang}.0-5000.txt'
            self.dico = load_dictionary(os.path.join(self.params.dico_eval_path, filename), word2id1, word2id2)
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        self.dico = self.dico.to(self.params.device)

    def build_translation_dictionary(self):
        src_emb = self.mapping(self.src_emb.weight).detach()
        tgt_emb = self.tgt_emb.weight.detach()
        src_emb = nn.functional.normalize(src_emb, p=2, dim=1)
        tgt_emb = nn.functional.normalize(tgt_emb, p=2, dim=1)
        self.dico = build_translation_dictionary(src_emb, tgt_emb, self.params)

    def solve_procrustes(self):
        A = self.src_emb.weight[self.dico[:, 0]].cpu().numpy()
        B = self.tgt_emb.weight[self.dico[:, 1]].cpu().numpy()
        W = self.mapping.weight.data.cpu().numpy()
        M = B.T @ A
        U, _, V_t = scipy.linalg.svd(M, full_matrices=True)
        W = U @ V_t
        self.mapping.weight.data.copy_(torch.from_numpy(W).to(self.params.device))

    def orthogonalize(self):
        if self.params.map_beta > 0:
            W = self.mapping.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.t()).mm(W))

    def save_best_model(self):
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info(f'* Saving the best model to {path}...')
        torch.save(self.mapping.state_dict(), path)

    def reload_best_model(self):
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info(f'* Reloading the best model from {path}...')
        assert os.path.isfile(path)
        self.mapping.load_state_dict(torch.load(path,weights_only=True))

    def export_embeddings(self):
        params = self.params
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        src_emb = self.mapping(src_emb.to(params.device)).data.cpu()
        export_embeddings(src_emb, tgt_emb, params)