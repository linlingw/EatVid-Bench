# EatVid-Bench 上传指南

## ✅ 目录已整理

```
release/
├── README.md                    # 项目说明（含 HuggingFace 链接）
├── LICENSE                      # CC BY-NC 4.0
├── requirements.txt            # Python 依赖
├── .gitignore                  # Git 忽略规则
│
├── docs/
│   ├── DATASET.md             # 数据集说明
│   └── EVALUATION.md          # 评测协议
│
├── eatvid_benchmark/          # 主代码包
│   ├── __init__.py
│   ├── run_qa_baseline.py     # 主评测脚本 ✅
│   ├── qa_evaluation/         # 评测核心
│   │   ├── __init__.py
│   │   ├── qa_dataset.py
│   │   ├── qa_metrics.py
│   │   ├── qa_task.py
│   │   ├── qa_prompts.py
│   │   └── qa_prompts_advanced.py
│   ├── models/                # 模型接口
│   │   ├── baseline_models.py
│   │   └── opensource/       # 开源模型实现
│   ├── data/
│   │   └── split.py
│   └── utils/
│       └── gpt_judge.py       # GPT 裁判
│
├── finetune/                  # 微调工具（已清理）
│   ├── generate_cot.py
│   ├── prepare_train_data.py
│   ├── evaluate.py
│   └── README.md
│
├── configs/
│   └── experiment_config.yaml
│
└── examples/
    └── run_evaluation.sh
```

## 🚀 上传步骤

### 方法一：脚本上传

```bash
# 1. 配置 Git
git config --global user.email "linlingw@users.noreply.github.com"
git config --global user.name "linlingw"

# 2. 设置 Token
export GITHUB_TOKEN='ghp_xxxxxxxx'  # 替换为你的实际 token

# 3. 运行脚本
bash /media/nas_data/zxp_data/eccv/scripts/github_upload.sh
```

### 方法二：手动上传

```bash
cd /media/nas_data/zxp_data/eccv/release

# 初始化
git init
git config user.email "linlingw@users.noreply.github.com"
git config user.name "linlingw"

# 添加文件
git add .
git commit -m "Initial release of EatVid-Bench benchmark"

# 推送（替换 token）
git remote add origin https://你的token@github.com/linlingw/EatVid-Bench.git
git branch -M main
git push -u origin main
```

### 方法三：网页上传

1. 访问 https://github.com/linlingw/EatVid-Bench
2. 点击 "uploading an existing file"
3. 上传所有文件

## 📝 ECCV 表单

**第一栏（URL）：**
```
https://github.com/linlingw/EatVid-Bench
```

**第二栏（描述）：**
```
EatVid-Bench is a multimodal fine-grained eating behavior video benchmark. 
The released materials include structured annotations, multimodal QA samples, 
benchmark splits, metadata, evaluation protocols, and scripts for reproducing 
the benchmark evaluation. The HuggingFace repository provides the dataset files 
and annotation metadata, while the GitHub repository provides the benchmark code, 
data loading utilities, evaluation scripts, result submission format, metric 
computation, and documentation.
```
