# %%
import os




# %%
import pickle
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from functools import partial


from torch.utils.data import Dataset, DataLoader, 

# %%
import pytorch_lightning as pl

seed = 42
pl.seed_everything(seed)
torch.set_float32_matmul_precision("medium")


# %% [markdown]
# 两个feature，一个emb，一个结构

# %%
data_path = "saisdata/WSAA_data_test.pkl"

# %%
test_datas = pickle.load(open(os.path.join( data_path), "rb"))


# %%
class base_dataset(Dataset):

    def __init__(self, dict_data):
        RESTYPES = "ACDEFGHIKLMNPQRSTVWY"  # 定义允许的氨基酸类型

        cleaned_sequences = []
        valid_labels = []  # 存储与有效序列对应的标签

        for i, d in enumerate(dict_data):
            sequence = d.get("sequence")  # 使用 .get() 更安全，防止 key 不存在
            label = d.get("label")

            # 检查序列是否为有效的非空字符串
            if sequence is None or not isinstance(sequence, str) or sequence == "":
                print(f"Warning: Item {i} has an invalid or empty sequence. Skipping.")
                continue  # 跳过无效的条目

            # 清理序列：将不在 RESTYPES 中的字符替换为 'X'
            cleaned_seq = "".join(
                [char if char in RESTYPES else "X" for char in sequence]
            )

            cleaned_sequences.append(cleaned_seq)
            valid_labels.append(label)  # 存储与清理后的序列对应的标签

        # 断言清理后的序列和标签列表长度一致
        assert len(cleaned_sequences) == len(
            valid_labels
        ), "Sequence and label lists should have the same length after cleaning."

        self.sequences = cleaned_sequences
        self.labels = valid_labels  # 使用过滤后的标签列表

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        # 返回对应的标签和清理后的序列
        return self.labels[index], self.sequences[index]


dataset = base_dataset(test_datas)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from transformers import AutoTokenizer, EsmModel

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")

# %%
embedding_list = []

base_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # 因为形状不一致
with torch.no_grad():
    model.eval()
    model.to(device)
    for labels, sequences in base_dataloader:
        inputs = tokenizer(sequences[0], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(dim=0)
        embeddings = embeddings.detach().cpu().numpy()
        embedding_list.append(embeddings)

os.makedirs("feature")
esm_embeding_path = os.path.join( "feature", "esm_embeding.h5")

with h5py.File(esm_embeding_path, "w") as f:
    for i, tensor in enumerate(embedding_list):
        f.create_dataset(f"embed_{i}", data=tensor)

# %%
torch.cuda.empty_cache()

# %%
from transformers import AutoTokenizer, EsmForProteinFolding

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

# %%
torch.backends.cuda.matmul.allow_tf32 = True
model.esm = model.esm.half()
model.trunk.set_chunk_size(64)


# %%
def chunked_collate_valid(batch, max_seq_len: int = 1024, step: int = 64):
    """Slice long sequences into (possibly overlapping) chunks and assemble a minibatch.

    Returns a dict with keys::

        input_ids:     LongTensor [n_chunks, L]
        attention_mask:BoolTensor  [n_chunks, L]
        chunk_labels:  LongTensor [n_chunks, L]  (present only when labels given)
    """

    batch_seq_list, seq_lengths, batch_indices = (
        [],
        [],
        [],
    )

    for batch_idx, (lbl, seq) in enumerate(batch):
        seq_len = len(seq)

        start = 0
        while start < seq_len:

            end = min(start + max_seq_len, seq_len)

            per_batch_seq = seq[start:end]

            batch_seq_list.append(per_batch_seq)
            batch_indices.append((batch_idx, start, end))
            seq_lengths.append(end - start)
            # 只要到了终点就不再继续
            if seq_len == end:
                break
            start += step

    return (
        batch_seq_list,
        seq_lengths,
        batch_indices,
    )


from functools import partial

max_seq_len = 512
step = int(max_seq_len / 2)
collate_fn_valid = partial(chunked_collate_valid, max_seq_len=max_seq_len, step=step)

# %%
stru_fea_list = []

base_dataloader = DataLoader(
    dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_valid
)  # 因为形状不一致
with torch.no_grad():
    model.eval()
    model.to(device)
    for batch_seq_list, seq_lengths, batch_indices in base_dataloader:
        # 获取当前批次所有序列的总长度
        total_length = batch_indices[-1][-1]

        # 创建两个数组：一个用于累加特征值，一个用于累加计数
        # 特征值累加数组使用浮点类型
        seq_preds_sum = np.zeros(shape=(total_length, 2), dtype=np.float32)
        # 计数数组使用整数类型
        seq_preds_count = np.zeros(shape=(total_length, 2), dtype=np.int32)

        for i, ((batch_idx, start, end), seq_length, batch_seq) in enumerate(
            zip(batch_indices, seq_lengths, batch_seq_list)
        ):

            inputs = tokenizer(batch_seq, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            # angle_rep = (
            #     outputs.angles[-1, 0, :, :3, :]
            #     .reshape(outputs.angles.shape[2], -1)
            #     .detach()
            #     .cpu()
            #     .numpy()
            # ) # [L ,6]
            mean_plddt = (
                torch.gather(
                    outputs.plddt,
                    2,
                    outputs.aatype.unsqueeze(-1),
                )
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            mean_pae = (
                torch.mean(outputs.predicted_aligned_error.squeeze(0), dim=-1)
                + torch.mean(outputs.predicted_aligned_error.squeeze(0), dim=-2)
            ).unsqueeze(1).detach().cpu().numpy() / 2

            full_seq_fea = np.concatenate((mean_plddt, mean_pae), axis=-1)
            current_slice = slice(start, end)
            seq_preds_sum[current_slice, :] += full_seq_fea
            seq_preds_count[current_slice, :] += 1

        seq_preds_avg = seq_preds_sum / seq_preds_count
        if np.isnan(seq_preds_avg).any():
            break

        stru_fea_list.append(seq_preds_avg)

# %%
stru_embeding_path = os.path.join("feature", "stru_embeding.h5")

with h5py.File(stru_embeding_path, "w") as f:
    for i, tensor in enumerate(stru_fea_list):
        f.create_dataset(f"embed_{i}", data=tensor)

# %%
