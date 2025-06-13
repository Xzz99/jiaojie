# full_pipeline.py

import os
import re
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# ===================== 氨基酸属性字典 ===================== #
amino_acid_properties = {
    'A': [1.8, 91.5, 0, 0, 15, -0.5, 27.8, 4.34, 705.42, 31.5, 70.079, 0.48],
    'C': [2.5, 118, 1.48, 8.18, 5, -1, 15.5, 35.77, 2412.56, 13.9, 103.144, 0.32], 
    'D': [-3.5, 124.5, 40.7, 3.65, 50, 3, 60.6, 12, 34.96, 60.9, 115.089, 0.81],
    'E': [-3.5, 115.1, 49.91, 4.25, 55, 3, 68.2, 17.26, 1158.66, 72.3, 129.116, 0.93],
    'F': [2.8, 203.4, 0.35, 0, 10, -2.5, 25.5, 29.4, 5203.86, 28.7, 147.177, 0.42],
    'G': [-0.4, 66.4, 0, 0, 10, 0, 24.5, 0, 33.18, 25.2, 57.052, 0.51],
    'H': [-3.2, 167.3, 51.6, 6, 34, -0.5, 50.7, 21.81, 1637.13, 46.7, 137.142, 0.66],
    'I': [4.5, 168.8, 0.15, 0, 13, -1.8, 22.8, 19.06, 5979.4, 23, 113.16, 0.39],
    'K': [-3.9, 171.3, 49.5, 10.53, 85, 3, 103, 21.29, 699.69, 110.3, 128.174, 0.93],
    'L': [3.8, 167.9, 0.45, 0, 16, -1.8, 27.6, 18.78, 4985.7, 29, 113.16, 0.41],
    'M': [1.9, 170.8, 1.43, 0, 20, -1.3, 33.5, 21.64, 4491.66, 30.5, 131.198, 0.44],
    'N': [-3.5, 135.2, 3.38, 0, 49, 0.2, 60.1, 13.28, 513.46, 62.2, 114.104, 0.82],
    'P': [-1.6, 129.3, 1.58, 0, 45, 0, 51.5, 10.93, 431.96, 53.7, 97.177, 0.78],
    'Q': [-3.5, 161.1, 3.53, 0, 56, 0.2, 68.7, 17.56, 1087.83, 74, 128.131, 0.81], 
    'R': [-4.5, 202, 52, 12.48, 67, 3, 94.7, 26.66, 1484.28, 93.8, 156.188, 0.84],
    'S': [-0.8, 99.1, 1.67, 0, 32, 0.3, 42, 6.35, 174.76, 44.2, 87.078, 0.7], 
    'T': [-0.7, 122.1, 1.66, 0, 32, -0.4, 45, 11.01, 601.88, 46, 101.105, 0.71],
    'V': [4.2, 141.7, 0.13, 0, 14, -1.5, 23.7, 13.92, 4474.4, 23.5, 99.133, 0.4],
    'W': [-0.9, 237.6, 2.1, 0, 17, -3.4, 34.7, 42.53, 6374.07, 41.7, 186.213, 0.49],
    'Y': [-1.3, 203.6, 1.61, 10.7, 41, -2.3, 55.2, 31.53, 4291.1, 59.1, 163.17, 0.67],
    'X': [0]*12
}

# ===================== PseAAC 计算函数 ===================== #
def compute_pse_aac(sequence, k=2, lambda_=5):
    features = []
    for i in range(12):
        aac = [amino_acid_properties.get(aa, [0]*12)[i] for aa in sequence]
        theta = [
            np.mean([
                (amino_acid_properties.get(sequence[j], [0]*12)[i] - amino_acid_properties.get(sequence[j+lam], [0]*12)[i])**2
                for j in range(len(sequence)-lam)
            ]) if len(sequence) > lam else 0
            for lam in range(1, lambda_+1)
        ]
        features.extend(aac + theta)
    features = np.array(features, dtype=np.float32)
    norm = np.linalg.norm(features)
    return features / norm if norm != 0 else features

# ===================== Cross-Attention 网络结构 ===================== #
class CrossAttentionModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, pseaac_dim, protbert_dim, num_classes, dropout=0.5):
        super(CrossAttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pseaac_projection = nn.Linear(pseaac_dim, embedding_dim)
        self.protbert_projection = nn.Linear(protbert_dim, embedding_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * embedding_dim, embedding_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, pseaac_features, protbert_features):
        pseaac_features = self.pseaac_projection(pseaac_features)
        protbert_features = self.protbert_projection(protbert_features)
        pseaac_features = pseaac_features.unsqueeze(0)
        protbert_features = protbert_features.unsqueeze(0)
        attn_output, _ = self.cross_attention(query=pseaac_features, key=protbert_features, value=protbert_features)
        for _ in range(self.num_layers):
            self_attn_output, _ = self.self_attention(query=attn_output, key=attn_output, value=attn_output)
            attn_output = self.layer_norm1(attn_output + self_attn_output)
            ff_output = self.feed_forward(attn_output)
            attn_output = self.layer_norm2(attn_output + ff_output)
        logits = self.classifier(attn_output.squeeze(0))
        return logits

# ===================== 主流程 ===================== #
if __name__ == '__main__':
    fasta_path = "./test.fasta"
    h5_output_path = "./feature/test.h5"
    csv_output_path = "./feature/test.csv"
    sequence_txt_path = "./test.csv"
    model_weights_path = "./best_model_e20p5.pth"
    pos_out = "./pre_positive.csv"
    neg_out = "./pre_negative.csv"

    # Step 1: BERT 提取
    tokenizer = BertTokenizer.from_pretrained("./newnewnew_p3e110(2)", do_lower_case=False)
    model = BertModel.from_pretrained("./newnewnew_p3e110(2)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model).to(device)

    protbert_features, seq_ids, seq_strs, pseaac_features = [], [], [], []

    for record in tqdm(SeqIO.parse(fasta_path, "fasta")):
        seq_id, seq = record.id, str(record.seq)
        cleaned_seq = re.sub(r"(?<=\w)(?=\w)", " ", re.sub(r"[UZOB]", "X", seq))
        tokenized = tokenizer(cleaned_seq, return_tensors='pt').to(device)
        with torch.no_grad():
            embedding = model(**tokenized)[1].detach().cpu().numpy()[0]
        protbert_features.append(embedding)
        padded_seq = seq + 'X' * (30 - len(seq))
        pseaac_features.append(compute_pse_aac(padded_seq))
        seq_ids.append(seq_id)
        seq_strs.append(seq)

    pd.DataFrame(protbert_features, index=seq_ids).to_hdf(h5_output_path, key='data', mode='a', complevel=4, complib='blosc')
    pd.DataFrame(pseaac_features).to_csv(csv_output_path, index=False)
    pd.DataFrame({"Sequence": seq_strs}).to_csv(sequence_txt_path, index=False)

    # Step 2: 预测
    test_pseaac = torch.tensor(pd.read_csv(csv_output_path).values, dtype=torch.float32)
    test_protbert = torch.tensor(pd.read_hdf(h5_output_path).values, dtype=torch.float32)
    sequences = pd.read_csv(sequence_txt_path)["Sequence"].tolist()
    model_attn = CrossAttentionModel(256, 4, 2, 420, 1024, 2).to(device)
    model_attn.load_state_dict(torch.load(model_weights_path))
    model_attn.eval()

    test_dataset = TensorDataset(test_pseaac, test_protbert)
    test_loader = DataLoader(test_dataset, batch_size=32)

    positive_samples, negative_samples = [], []
    with torch.no_grad():
        for batch_idx, (pseaac_batch, protbert_batch) in enumerate(tqdm(test_loader, desc="Testing")):
            pseaac_batch, protbert_batch = pseaac_batch.to(device), protbert_batch.to(device)
            outputs = model_attn(pseaac_batch, protbert_batch)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            start_idx = batch_idx * 32
            end_idx = start_idx + len(predictions)
            batch_sequences = sequences[start_idx:end_idx]
            for i in range(len(predictions)):
                sample_info = {
                    "Sequence": batch_sequences[i],
                    "Class_0_Prob": probabilities[i][0],
                    "Class_1_Prob": probabilities[i][1],
                }
                (positive_samples if predictions[i] == 1 else negative_samples).append(sample_info)

    pd.DataFrame(positive_samples).to_csv(pos_out, index=False)
    pd.DataFrame(negative_samples).to_csv(neg_out, index=False)
    print("预测完成，结果已保存")
