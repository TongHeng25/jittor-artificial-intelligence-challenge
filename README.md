# 第五届计图人工智能挑战赛-赛题二人体骨骼生成

## 📌 项目概述
本项目主要面向人体建模中的两个核心任务：  

1. **皮肤变形权重预测**  
   - 输入：人体网格顶点坐标 + 关节位置  
   - 输出：每个顶点对 52 个关节的变形权重  

2. **人体关节位置预测**  
   - 输入：人体点云数据  
   - 输出：52 个关节的 3D 坐标  

---

## 💡 解决思路

### 点云编码器
- 使用 **PointTransformerVAE** 对点云进行预训练编码  
- 编码器-解码器结构 (VAE) 学习点云全局特征  
- 提取的全局特征可增强关节位置预测和权重预测的性能  

### 人体关节位置预测
- **输入**：点云数据  
- **输出**：52 个关节的 3D 坐标  
- **核心架构**：
  - DGCNN：两层 EdgeConv 提取点云局部特征  
  - PointTransformerVAE：提供全局特征  
  - Transformer 块：融合局部与全局特征  
  - 双头输出：
    - 关节权重头：预测点对关节的权重  
    - 位置回归头：预测点的 3D 坐标  
    - 最终关节位置 = 权重加权平均  

### 皮肤变形权重预测
- **输入**：人体网格顶点 + 关节位置  
- **输出**：每个顶点对 52 个关节的权重  
- **核心架构**：
  - EdgeConv：提取顶点局部几何特征  
  - Joint Encoder：编码关节位置  
  - 相对位置编码：建模顶点-关节间关系  
  - Transformer 融合：多头注意力融合顶点特征、关节特征和点云全局特征  
  - 权重预测：输出每个顶点的权重分布  

---

## 📂 代码结构
jittor-comp-human-main/
├── model/                    # 核心模型实现
├── models/                   # 模型工厂和接口
├── dataset/                  # 数据处理模块
├── PCT/                      # Point Cloud Transformer 库
├── data/                     # 数据集
├── output/                   # 训练输出
├── predict/                  # 预测结果
└── \*.py                      # 训练和预测脚本


### 核心模型文件
- **udt.py**  
  - PointTransformerVAE：点云 VAE 预训练模型  
  - 提供点云全局特征先验（训练目标：重构损失 + KL 散度）  

- **skin.py**  
  - Our_transfomer：皮肤变形权重预测模型  
  - 输入：网格顶点 + 关节位置  
  - 输出：每个顶点对 52 个关节的权重  
  - 模块：EdgeConv、JointEncoder、TransformerBlock、PointTransformerVAE  

- **skeleton.py**  
  - OurTransformerNet：人体关节位置预测模型  
  - 输入：点云  
  - 输出：52 个关节的 3D 坐标  
  - 模块：两层 EdgeConv、TransformerLayer（自注意力 + 交叉注意力）、双头输出（权重 + 位置回归）  

---

## ⚙️ 环境配置

- 系统：Ubuntu 22.04.4 LTS  
- CUDA：11.7  
- Python：3.9.23  

创建 Conda 环境：
```bash
conda env create -f environment.yml
conda activate jittor_comp_human
```

---

## 🚀 使用说明

### 关节位置预测

```bash
# 使用预训练模型预测
python predict_skeleton.py

# 使用训练好的模型预测
python predict_skeleton.py --pretrained_model output/skeleton_new/best_model.pkl
```

### 皮肤权重预测

```bash
# 使用预训练模型预测
python predict_skin.py

# 使用训练好的模型预测
python predict_skin.py --pretrained_model output/skin_new/best_model.pkl
```

### 数据增强

```bash
python save_aug_dataset.py   # 请先修改数据集路径
# 增强数据将保存至 data/train_aug
```

### 模型训练

```bash
# 训练关节预测模型
python train_skeleton.py

# 训练皮肤权重预测模型
python train_skin.py
```


