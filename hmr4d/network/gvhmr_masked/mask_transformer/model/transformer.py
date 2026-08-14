import torch
import torch.nn as nn
import numpy as np
import einops

import math
from torch.distributions.categorical import Categorical

from .tools import *
from .gvhmr_conditioned_transformer import GVHMRConditionedTransformer


class MaskTransformer(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super(MaskTransformer, self).__init__()
        """
        Preparing Networks
        Wrap DSTFormer
        """
        self.cfg = cfg
        self.task = cfg.task 

        self.dst_former = GVHMRConditionedTransformer(cfg.dst_former)

        self.num_tokens = cfg.num_tokens

        self.dim_token = cfg.dim_token
        self.num_codes = cfg.num_codes
        _num_codes = (
            self.num_codes + 1
        )  # dummy tokens for masking
        self.mask_id = self.num_codes

        self.token_emb = nn.Embedding(_num_codes, self.dim_token)

        self.noise_schedule = cosine_schedule
        self.mask_tau = cfg.mask_tau
        self.scheme_prob = cfg.scheme_prob
        self.full_label = cfg.full_label

    def load_and_freeze_token_emb(self, codebook):
        """
        Preparing frozen weights
        :param codebook: (c, d)
        :return:
        """
        # initialization
        self.token_emb.weight = nn.Parameter(
            torch.cat(
                [codebook, torch.zeros(size=(1, codebook.shape[1]), device=codebook.device)], dim=0
            )
        )  # add zero token for mask
        self.token_emb.requires_grad_(False)
        print("Token embedding initialized!")

    def trans_forward(self, masked_ids, batch, **kwargs):
        """
        :param ids: (B, F, J)
        :param cond: images, (B, F, J_i, C), J_i = H_i * W_i
        :return:
            logits: (B, C, F, J)
        """
        x = self.token_emb(masked_ids)
        out = self.dst_former(x, batch, **kwargs)
        return out

    def test_forward(
            self, 
            batch, 
            ids, 
            last_output=None, 
            cond_scale=1.
            ):
        tokens = self.token_emb(ids)
        out = self.dst_former.inference(tokens, batch, last_output=last_output)
        if self.task != "video" or cond_scale == 1:
            return out
        
        aux_out = self.dst_former(tokens, batch, cond_out=out)

        logits = out["logits"]
        aux_logits = aux_out["logits"]
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        out["logits"] = scaled_logits
        return out

    def get_random_mask(self, ids, stage="train", key_padding_mask=None):
        B, F, J = ids.shape
        N = F * J
        if key_padding_mask is None:
            valid_mask = torch.ones((B, N), device=ids.device, dtype=torch.bool)
        else:
            valid_mask = (~key_padding_mask[..., None]).expand(-1, -1, J).reshape(B, N)
        num_valid_tokens = valid_mask.sum(-1)

        if stage == "train":
            rand_time = uniform((B,), device=ids.device) * self.mask_tau
            rand_mask_probs = self.noise_schedule(rand_time)
            num_token_masked = (num_valid_tokens * rand_mask_probs).round().clamp(min=1)

            batch_randperm = torch.rand((B, N), device=ids.device)
            batch_randperm = batch_randperm.masked_fill(~valid_mask, float("inf"))
            batch_randperm = batch_randperm.argsort(dim=-1).argsort(dim=-1)
            mask = (batch_randperm < num_token_masked.unsqueeze(-1)) & valid_mask
            mask = mask.view(B, F, J)
        elif stage in ["val", "test"]:
            mask_ratio = self.cfg.get("mask_ratio", 0.2)
            num_token_masked = (num_valid_tokens * mask_ratio).round()

            batch_randperm = torch.rand((B, N), device=ids.device)
            batch_randperm = batch_randperm.masked_fill(~valid_mask, float("inf"))
            batch_randperm = batch_randperm.argsort(dim=-1).argsort(dim=-1)
            mask = (batch_randperm < num_token_masked.unsqueeze(-1)) & valid_mask
            mask = mask.view(B, F, J)
        else:
            raise NotImplementedError("Unsupported stage!!!")

        return mask

    def get_full_mask(self, ids, stage="train"):
        B, F, J = ids.shape
            
        mask = torch.ones((B, F, J), device=ids.device, dtype=torch.bool)

        return mask

    def parse_mask_scheme(self, mask_scheme, step):
        mode = None
        for scheme in mask_scheme:
            end_step = scheme[0]
            if step < end_step or end_step < 0:
                mode = scheme[1]
                break
        if mode is None:
            raise ValueError("Invalid mask scheme!!!")
        
        prob_mode = self.scheme_prob 
        if "+" in mode:
            possible_modes = mode.split("+")
            prob = [prob_mode[m] for m in possible_modes]
            prob = np.array(prob) / np.sum(prob)
            mode = np.random.choice(possible_modes, p=prob)
        return mode

    def get_mask(self, ids, stage="train", mode=None, step=0, key_padding_mask=None):
        """
        :param ids: (B, F, J)
        :param stage: "train" or "test"
        :param mode: "random", "full", "inference"
        :return:
        """
        if mode == "random":
            mask = self.get_random_mask(ids, stage=stage, key_padding_mask=key_padding_mask)
        elif mode == "full" or mode == "inference":
            mask = self.get_full_mask(ids, stage=stage)
        else:
            raise NotImplementedError("Unsupported mask mode!!!")

        return mask, mode

    @torch.no_grad()
    def get_inference_mask(self, batch, ratio=None):
        """
        Generate one step inference tokens for training
        Start from all tokens being masked and recover for one step
        """
        ids = batch["ids"]
        B, F, J = ids.shape
        N = F * J
        key_padding_mask = batch.get("key_padding_mask", None)
        if key_padding_mask is None:
            valid_mask = torch.ones((B, N), device=ids.device, dtype=torch.bool)
        else:
            valid_mask = (~key_padding_mask[..., None]).expand(-1, -1, J).reshape(B, N)
        num_valid_tokens = valid_mask.sum(-1)

        masked_ids = torch.full_like(ids, fill_value=self.mask_id)
        masked_ids = masked_ids.view(B, N)

        if ratio is None:
            rand_time = uniform((B,), device=ids.device) * self.mask_tau
            rand_mask_probs = self.noise_schedule(rand_time)
        else:
            rand_mask_probs = torch.full((B,), fill_value=ratio, device=ids.device)
        num_token_masked = (num_valid_tokens * rand_mask_probs).round().clamp(min=1)

        out = self.trans_forward(masked_ids.view(B, F, J), batch)
        logits = out["logits"]

        logits = einops.rearrange(logits, "b c f j -> b (f j) c")  # (B, N, C)
        # use argmax to pick the most probable token
        pred_ids = logits.argmax(dim=-1)  # (B, N)
        probs = logits.softmax(dim=-1)  # (B, N, C)
        scores = probs.gather(2, pred_ids.unsqueeze(dim=-1)).squeeze(-1)  # (B, N)
        scores = scores.masked_fill(~valid_mask, 1e5)
        sorted_indices = scores.argsort(dim=1) # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
        ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
        mask = (ranks < num_token_masked.unsqueeze(-1)) & valid_mask

        x_ids = torch.where(mask, self.mask_id, pred_ids)
        if self.full_label:
            labels = ids.view(B, N) # supervise all tokens
        else:
            labels = torch.where(mask, ids.view(B, N), self.mask_id)

        x_ids = x_ids.view(B, F, J)
        labels = labels.view(B, F, J)
        x_ids = x_ids.masked_fill(~valid_mask.view(B, F, J), self.mask_id)
        labels = labels.masked_fill(~valid_mask.view(B, F, J), self.mask_id)

        return x_ids, labels

    def training_step(self, batch, step=0):
        """
        Prepare mask
        """
        mask_scheme = self.cfg.mask_scheme
        mode = self.parse_mask_scheme(mask_scheme, step)

        if mode == "inference":
            x_ids, labels = self.get_inference_mask(batch)
            out = self.trans_forward(x_ids, batch)
        else:
            ids = batch["ids"]
            mask, mode = self.get_mask(
                ids,
                stage="train",
                mode=mode,
                step=step,
                key_padding_mask=batch.get("key_padding_mask", None),
            )

            # Note this is our training target, not input
            if self.full_label:
                labels = ids
            else:
                labels = torch.where(mask, ids, self.mask_id)

            x_ids = ids.clone()

            # Further Apply Bert Masking Scheme
            # Step 1: 10% replace with an incorrect token
            prob_rid = self.cfg.prob_rid if mode != "full" else 0.0
            # if self.cfg.rmid_random_only and mode != "random":
            #     prob_rid = 0.0
            mask_rid = get_mask_subset_prob(mask, prob_rid)
            rand_id = torch.randint_like(x_ids, high=self.num_codes)
            x_ids = torch.where(mask_rid, rand_id, x_ids)
            # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
            prob_mid = self.cfg.prob_mid if mode != "full" else 1.0
            # if self.cfg.rmid_random_only and mode != "random":
            #     prob_mid = 1.0
            mask_mid = get_mask_subset_prob(mask & ~mask_rid, prob_mid)

            x_ids = torch.where(mask_mid, self.mask_id, x_ids)
            out = self.trans_forward(x_ids, batch)
        logits = out["logits"]
        ce_loss, pred_ids, acc = cal_performance(
            logits, labels, ignore_index=self.mask_id, focal_gamma=self.cfg.loss.get("focal_gamma", 0.)
        )
        ret = {
            "ce_loss": ce_loss,
            "acc": acc,
            "pred_ids": pred_ids,
        }
        ret.update(out)

        return ret

    def generate(
        self,
        batch,
        masked_ids,
        timesteps: int,
        cond_scale: int,
        temperature=1,
        topk=1,
        gsample=False,
    ):
        """
        Generate pose sequence for test stage
        Start from all tokens being masked and iteratively recover tokens
        """

        B, F, J = masked_ids.shape
        N = F * J
        device = masked_ids.device

        ids = masked_ids.clone()
        ids = ids.view(B, N)
        key_padding_mask = batch.get("key_padding_mask", None)
        if key_padding_mask is None:
            valid_mask = torch.ones((B, N), device=ids.device, dtype=torch.bool)
        else:
            valid_mask = (~key_padding_mask[..., None]).expand(-1, -1, J).reshape(B, N)
        ids = ids.masked_fill(~valid_mask, 0)
        mask = (ids == self.mask_id) & valid_mask
        num_masked_init = mask.sum(dim=-1)
        scores = torch.zeros_like(ids, device=ids.device)
        scores = scores.masked_fill(~mask, 1e5)

        timesteps = timesteps * mask.sum() / valid_mask.sum().clamp(min=1)
        timesteps = timesteps.round().int().clamp(min=1).item()

        # Start from tokens
        starting_temperature = temperature

        # save the tokens at every timestep
        timestep_ids = []
        last_output = None

        for timestep, timestep_int in zip(
            torch.linspace(0, 1, timesteps + 1, device=device)[:timesteps], range(timesteps)
        ):
            # 0 < timestep < 1
            rand_mask_prob = (torch.cos(timestep * math.pi) + 1) * 0.5
            # rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            """
            Maskout, and cope with variable length
            """
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(
                rand_mask_prob * num_masked_init 
            ).clamp(
                min=1
            )  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(
                dim=1
            )  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(
                dim=1
            )  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1)) & valid_mask
            ids = torch.where(is_mask, self.mask_id, ids)
            timestep_ids.append(ids.clone())

            """
            Preparing input
            """
            out = self.test_forward(batch, ids.view(B, F, J), last_output=last_output, cond_scale=cond_scale)

            last_output = out
            logits = out["logits"]  # (B, C, F, J)

            logits = einops.rearrange(logits, "b c f j -> b (f j) c")  # (B, N, C)
            # clean low prob token
            # use greedy sampling
            filtered_logits = top_k(logits, k=topk, dim=-1)

            """
            Update ids
            """
            # if force_mask:
            temperature = starting_temperature
            # else:
            # temperature = starting_temperature * (steps_until_x0 / timesteps)
            # temperature = max(temperature, 1e-4)
            # print(filtered_logits.shape)
            # temperature is annealed, gradually reducing temperature as well as randomness
            if gsample:  # use gumbel_softmax sampling
                pred_ids = gumbel_sample(
                    filtered_logits, temperature=temperature, dim=-1
                )  # (B, N)
            else:  # use multinomial sampling
                probs = nn.functional.softmax(filtered_logits / temperature, dim=-1)  # (B, N, C)
                # print(temperature, starting_temperature, steps_until_x0, timesteps)
                # print(probs / temperature)
                pred_ids = Categorical(probs).sample()  # (B, N)

            ids = torch.where(is_mask, pred_ids, ids)

            """
            Updating scores
            """
            probs_without_temperature = logits.softmax(dim=-1)  # (B, N, C)
            scores = probs_without_temperature.gather(
                2, pred_ids.unsqueeze(dim=-1)
            )  # (B, N, 1)
            scores = scores.squeeze(-1)  # (B, N)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = ids.view(B, F, J)
        out.update({
            "masked_ids": masked_ids,
            "pred_ids": ids,
        }) 

        return out

