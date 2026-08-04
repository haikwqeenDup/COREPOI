from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class NextPOIDataset(Dataset):
    """
    Next-POI fine-tuning dataset.

    Each sample corresponds to a trajectory cut position:
      input_tokens: [max_len] with CLS + history, PAD at tail
      time features: tod/dow aligned to history (CLS/PAD -> 0), shifted by +1
      query time: tod_label/dow_label (for the NEXT step)
      target_poi: scalar token id (>=3)
      target_cat/target_regions: scalars derived from target_poi
    """

    def __init__(
        self,
        trajs: List[Tuple[int, List[int], List[int], List[int]]],
        *,
        token2cat: np.ndarray,
        token2regions: List[np.ndarray],
        max_len: int = 128,
        pad_token: int = 0,
        cls_token: int = 2,
        augment: bool = True,
    ):
        super().__init__()
        self.trajs = trajs
        self.token2cat = token2cat
        self.token2regions = token2regions
        self.max_len = int(max_len)
        self.pad_token = int(pad_token)
        self.cls_token = int(cls_token)
        self.augment = bool(augment)

        # index list of (traj_idx, cut_pos)
        self.cuts: List[Tuple[int, int]] = []
        max_hist = self.max_len - 1  # reserve CLS
        for ti, (_, tokens, tod, dow) in enumerate(self.trajs):
            L = len(tokens)
            if L <= 1:
                continue
            if self.augment:
                for cut in range(1, L):  # label at cut
                    self.cuts.append((ti, cut))
            else:
                self.cuts.append((ti, L - 1))

        if len(self.cuts) == 0:
            raise ValueError("No finetune samples constructed. Check data.")

    def __len__(self):
        return len(self.cuts)

    def __getitem__(self, idx: int):
        traj_idx, cut = self.cuts[idx]
        user, tokens, tod, dow = self.trajs[traj_idx]

        hist_tokens = tokens[:cut]
        hist_tod = tod[:cut]
        hist_dow = dow[:cut]

        label_token = tokens[cut]
        label_tod = tod[cut] if cut < len(tod) else -1
        label_dow = dow[cut] if cut < len(dow) else -1

        # truncate history to last (max_len-1)
        max_hist = self.max_len - 1
        if len(hist_tokens) > max_hist:
            hist_tokens = hist_tokens[-max_hist:]
            hist_tod = hist_tod[-max_hist:]
            hist_dow = hist_dow[-max_hist:]

        seq_tokens = [self.cls_token] + hist_tokens
        seq_tod = [0] + [t + 1 if t >= 0 else 0 for t in hist_tod]
        seq_dow = [0] + [d + 1 if d >= 0 else 0 for d in hist_dow]

        attn = [1] * len(seq_tokens)
        pad_len = self.max_len - len(seq_tokens)
        if pad_len > 0:
            seq_tokens = seq_tokens + [self.pad_token] * pad_len
            seq_tod = seq_tod + [0] * pad_len
            seq_dow = seq_dow + [0] * pad_len
            attn = attn + [0] * pad_len

        query_tod = label_tod + 1 if label_tod >= 0 else 0
        query_dow = label_dow + 1 if label_dow >= 0 else 0

        target_cat = int(self.token2cat[label_token])
        target_regions = [int(t2r[label_token]) for t2r in self.token2regions]

        return {
            "user": int(user),
            "input_tokens": torch.tensor(seq_tokens, dtype=torch.long),
            "tod": torch.tensor(seq_tod, dtype=torch.long),
            "dow": torch.tensor(seq_dow, dtype=torch.long),
            "attn_mask": torch.tensor(attn, dtype=torch.bool),
            "query_tod": torch.tensor(query_tod, dtype=torch.long),
            "query_dow": torch.tensor(query_dow, dtype=torch.long),
            "target_poi": torch.tensor(label_token, dtype=torch.long),
            "target_cat": torch.tensor(target_cat, dtype=torch.long),
            "target_regions": torch.tensor(target_regions, dtype=torch.long),  # [S]
        }
