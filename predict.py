# %%
import os


# %%
import pickle
import h5py
import numpy as np
import torch
import torch.nn as nn


from functools import partial


from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# %%
import pytorch_lightning as pl

seed = 42
pl.seed_everything(seed)
torch.set_float32_matmul_precision("medium")


# %%
class EmbedDataset(Dataset):

    def __init__(
        self,
        llm_emb_path,
        stru_fea_path,
    ):
        super().__init__()
        self.llm_emb_list = self.load_h5py(llm_emb_path)

        self.stru_fea_list = self.load_h5py(stru_fea_path)

    def load_h5py(self, file_path):
        data_list = []

        def get_numeric_id_from_key(key_string):
            return int(key_string.split("_")[-1])

        with h5py.File(file_path, "r") as f:
            all_keys = list(f.keys())  # Get keys as a list

            sorted_keys = sorted(all_keys, key=get_numeric_id_from_key)

            data_list = [
                torch.from_numpy(f[key].__array__()).float() for key in sorted_keys
            ]

        return data_list

    def __len__(self):
        return len(self.stru_fea_list)

    def __getitem__(self, idx):

        return (
            self.llm_emb_list[idx],
            self.stru_fea_list[idx],
        )


def chunked_collate_valid(batch, max_seq_len: int = 1024, step: int = 64):
    """Slice long sequences into (possibly overlapping) chunks and assemble a minibatch.

    Returns a dict with keys::

        input_ids:     LongTensor [n_chunks, L]
        attention_mask:BoolTensor  [n_chunks, L]
        chunk_labels:  LongTensor [n_chunks, L]  (present only when labels given)
    """

    token_tensors, lengths, seq_lengths, batch_indices = (
        [],
        [],
        [],
        [],
    )

    for batch_idx, (llm_emb, stru_fea) in enumerate(batch):

        seq_len = llm_emb.size(0)

        full_len_token = torch.concat([llm_emb, stru_fea], dim=-1)

        seq_lengths.append(seq_len)

        start = 0
        while start < seq_len:
            end = min(start + max_seq_len, seq_len)

            # token ids
            token_tensors.append(full_len_token[start:end])

            lengths.append(end - start)
            batch_indices.append((batch_idx, start, end))
            if end == seq_len:
                break
            start += step

    # pad variable‑length tensors
    input_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0)
    attention_mask = (
        torch.arange(input_ids.size(1))[None, :] < torch.tensor(lengths)[:, None]
    ).cpu()

    return (
        input_ids,
        attention_mask,
        seq_lengths,
        batch_indices,
    )


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        # 对 nn.Linear 层的权重和偏置进行初始化
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, SeqWiseGLUBase):
        # 对 SeqWiseGLUBase 的权重和偏置进行初始化
        nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)


class GLUBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLUBase, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear1(x) * torch.sigmoid(self.linear2(x))


class SwiGLU(GLUBase):
    def forward(self, x):
        return nn.functional.silu(self.linear1(x)) * self.linear2(x)


def approximate_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class GEGLU(GLUBase):
    def forward(self, x):
        return self.linear1(x) * approximate_gelu(self.linear2(x))


class CustomGELU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomGELU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return approximate_gelu(self.linear(x))


class SeqWiseGLUBase(nn.Module):
    def __init__(self, s, f):
        super(SeqWiseGLUBase, self).__init__()
        # 合并两组权重，形状为 (s, f, 2)
        self.weight = nn.Parameter(torch.empty(s, f, 2))
        # 合并两组偏置，形状为 (s, 2)
        self.bias = nn.Parameter(torch.empty(s, 2))

    def forward(self, x):
        # x: (b, s, f)
        # 计算两个线性变换，结果形状为 (b, s, 2)
        out = torch.einsum("bsf,sfk->bsk", x, self.weight) + self.bias  # 'sfk' 中的 k=2
        # 分离两个输出
        out1 = out[:, :, 0]  # (b, s)
        out2 = out[:, :, 1]  # (b, s)
        # 应用 GLU 机制
        output = out1 * torch.sigmoid(out2)
        # (b, s)
        return output


class FC(nn.Module):
    def __init__(
        self,
        input_dim,
        linear_dims,
        fc_dropout,
        activation="swiglu",
        dropout_before_last=True,
    ):
        super(FC, self).__init__()

        if not linear_dims:
            raise ValueError("linear_dims should contain at least one layer dimension.")

        self.ACTIVATION_FUNCTIONS = {
            "swiglu": SwiGLU,
            "glu": GLUBase,
            "gelu": CustomGELU,
            "geglu": GEGLU,
        }

        activation_class = self.ACTIVATION_FUNCTIONS.get(activation)
        if activation_class is None:
            raise ValueError(f"Unsupported activation type: {activation}")

        layers = []
        for i, (prev_dim, mid_dim) in enumerate(
            zip([input_dim] + linear_dims[:-1], linear_dims)
        ):
            if fc_dropout > 0 and (dropout_before_last or i < len(linear_dims) - 1):
                layers.append(nn.Dropout(fc_dropout))
            layers.append(activation_class(prev_dim, mid_dim))

        self.fc_layers = nn.Sequential(*layers)
        self.fc_layers.apply(initialize_weights)

    def forward(self, x):
        return self.fc_layers(x)


# %%
from x_transformers import Decoder


class SimpleTransformer(pl.LightningModule):

    def __init__(
        self,
        input_dim: int = 20,
        d_model: int = 128,
        n_head: int = 3,
        n_layer: int = 3,
        layer_dropout: float = 0.3,
        seq_dropout: float = 0.3,
        pos_emb: str = "rope",
        gate_residual: bool = True,
        ff_swish=True,
        attn_gate_values=True,
        mid_dim=3,
        lr: float = 1e-6,
        wd: float = 0.1,
    ):
        super().__init__()

        encoder_params = {
            "use_simple_rmsnorm": True,
            "attn_flash": True,
            "ff_swish": ff_swish,
            "ff_glu": True,
            "gate_residual": gate_residual,
            "attn_gate_values": attn_gate_values,
        }
        if pos_emb == "rope":
            encoder_params["rotary_pos_emb"] = True

        self.embed = nn.Linear(input_dim, d_model)

        heads = 2**n_head
        self.trans = Decoder(
            dim=d_model,
            depth=n_layer,
            heads=heads,
            attn_dim_head=int(d_model / heads),
            layer_dropout=layer_dropout,
            attn_dropout=seq_dropout,
            ff_dropout=seq_dropout,
            **encoder_params,
        )
        mid_dim = [2] if mid_dim < 3 else [2**mid_dim, 2]
        self.local_sum_up = FC(
            d_model,
            mid_dim,
            seq_dropout,
            "glu",
            dropout_before_last=False,
        )
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([0.7 / 0.3]), reduction="none"
        )

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        x = self.trans(x, mask=attention_mask)
        x = self.local_sum_up(x)

        return x


# %%
max_seq_len = 512
step = int(max_seq_len / 2)
collate_fn_valid = partial(chunked_collate_valid, max_seq_len=max_seq_len, step=step)

# %%
llm_emb_path = os.path.join("feature", "esm_embeding.h5")
stru_fea_path = os.path.join("feature", "stru_embeding.h5")

dataset = EmbedDataset(llm_emb_path, stru_fea_path)

batch_size = 1
num_workers = 2
valid_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn_valid,
    num_workers=num_workers,
)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
import yaml


def load_model_yaml(file_path):
    with open(os.path.dirname(file_path) + "/hparams.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config


# %%
fold_pre = []
for i in range(5):
    fold_model_path = os.path.join("best_models", f"fold_{i}", "best_model.ckpt")
    model = SimpleTransformer.load_from_checkpoint(
        fold_model_path, **load_model_yaml(fold_model_path), map_location=device
    )
    model.eval()
    model_pre = []
    with torch.no_grad():
        for (
            input_ids,
            attention_mask,
            seq_lengths,
            batch_indices,
        ) in valid_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pred = model(input_ids, attention_mask)
            pred = torch.sigmoid(pred)
            local_pred = pred.cpu().numpy()

            seq_preds_sum = [
                torch.zeros(length, 2, device="cpu") for length in seq_lengths
            ]
            seq_counts = [
                torch.zeros(length, device="cpu", dtype=torch.float)
                for length in seq_lengths
            ]
            for i, (batch_idx, start, end) in enumerate(batch_indices):
                # 提取 chunk 的 logits 或概率，形状为 (end - start, num_classes)
                chunk_pred_scores = local_pred[i, : end - start, :]

                # 在对应位置累加 logits 或概率
                seq_preds_sum[batch_idx][start:end, :] += chunk_pred_scores
                # 在对应位置累加覆盖次数
                seq_counts[batch_idx][start:end] += 1

            total_f1 = 0.0  # 初始化总 F1

            for idx in range(len(seq_lengths)):

                # 获取该序列合并后的 logits 或概率总和
                seq_sum_scores = seq_preds_sum[idx]
                # 获取该序列的覆盖次数
                seq_counts_scores = seq_counts[idx].unsqueeze(
                    -1
                )  # 扩展维度以便广播相除

                # 计算平均 logits 或概率
                # 使用 torch.where 避免除以零，对于未覆盖的位置，其 score 将是 0（初始化值）
                seq_avg_scores = (
                    torch.where(
                        seq_counts_scores > 0,
                        seq_sum_scores / seq_counts_scores,
                        torch.zeros_like(
                            seq_sum_scores
                        ),  # 未覆盖位置可以保持为0或者用其他策略
                    )
                    .cpu()
                    .numpy()
                )
                model_pre.append(seq_avg_scores)
    fold_pre.append(model_pre)

# %%
final_predictions_list = []
n_samples = len(fold_pre[0])
for sample_idx in range(n_samples):
    sample_preds_across_folds = [
        fold_pre[fold_idx][sample_idx] for fold_idx in range(5)
    ]
    sample_preds_np = np.array(sample_preds_across_folds)
    mean_pred_sample = np.mean(sample_preds_np, axis=0)
    predicted_class_index = np.argmax(mean_pred_sample, axis=-1)
    final_predictions_list.append(predicted_class_index)

# %%
data_path = "saisdata/WSAA_data_test.pkl"
data = pickle.load(open(os.path.join(data_path), "rb"))

# %%
import pandas as pd

datadf = pd.DataFrame(data)

# %%
datadf["label"] = pd.Series(final_predictions_list)

# %%
datadf.to_csv("/saisresult/submit.csv", index=None)
