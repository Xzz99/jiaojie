# 环境导入

./模型/environment.yml

conda env create -f environment.yml



# 数据集

**dataset**中**new_data**和**Yan_data**

包含训练集验证集和测试集

# 微调过程

运行**finetune.sh**

其中预训练模型在**prot_bert_bfd**文件夹

微调后模型为**newnewnew_p3e110(2)**需要在链接https://pan.quark.cn/s/c77aa85e4481下载

# 特征提取

## 序列特征提取

运行**feature_extract_test2.py**

设置微调模型路径 数据集路径 输出序列特征路径

分别进行train/val/test特征提取

输出特征保存在**p3e110(2)**文件夹中new和Yan文件夹下.h5文件

## 理化特征提取

运行**PseAAC_1.py**

设置数据集路径 输出理化特征路径

分别进行train/val/test特征提取

输出特征保存在**PseAAC**文件夹中new和Yan文件夹下csv文件

# 特征融合及分类

运行**attention1.py**

设置特征路径 以及最佳模型保存路径

输出预测性能指标



# 第四章

## 数据

处理并去冗余序列保存在**4**/**data**/**smorf_protein1_0.9.fasta**

## 特征提取

运行**feature_extract_4.py**  和 **PseAAC_1_4.py**

特征保存在4/feature中

## 特征融合及分类

运行**attention1_4.py**

输出预测结果

正样本保存在**4/data/pre_positive.csv**

负样本保存在**4/data/pre_negative.csv**



## MIC值预测

运行**4/apex/predict.py**

预测结果保存在**4/apex/pre_positive_MICs.csv**

## 环境分析

[GitHub - BigDataBiology/SantosJunior_Torres_2024_AMPSphere_v1: Figures and files used in the AMPSphere manuscript](https://github.com/BigDataBiology/SantosJunior_Torres_2024_AMPSphere_v1)

预测序列的GMSC号保存在**4/data/PPsAMP_GMSC.fasta**

**./图表/sAMP_environment_distribution.csv**

## 属

**./图表/aquatic_genus_distribution.csv**

## KEGG注释分析

[eggNOG-mapper](http://eggnog-mapper.embl.de/job_status?jobname=MM_pe0_t97q)

**./图表/KEGG.xlsx**

