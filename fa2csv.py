import argparse
from Bio import SeqIO
import pandas as pd

def fasta_to_csv(fasta_file, csv_file):
    """
    将 FASTA 文件转换为 CSV 文件，列名为 Sequence
    Args:
        fasta_file: 输入的 FASTA 文件路径
        csv_file: 输出的 CSV 文件路径
    """
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    df = pd.DataFrame(sequences, columns=["Sequence"])
    df.to_csv(csv_file, index=False)
    print(f"转换完成，结果已保存到: {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将FASTA格式转换为带Sequence列的CSV文件")
    parser.add_argument('--fasta', required=True, help='输入FASTA文件路径')
    parser.add_argument('--csv', required=True, help='输出CSV文件路径')
    args = parser.parse_args()

    fasta_to_csv(args.fasta, args.csv)
