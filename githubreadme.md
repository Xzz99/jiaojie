# 1.配置环境

conda env create -f environment.yml

# 2.加载微调模型

training_args.bin

# 3.输入序列输出预测结果

python zhenghe.py --fasta ./your.fasta --seqcsv ./your.csv

其中输入序列fasta文件以及列名为Sequence的csv文件，这里给出fasta转换为csv格式代码

python fasta_to_csv.py --fasta ./your.fasta --csv ./your.csv

