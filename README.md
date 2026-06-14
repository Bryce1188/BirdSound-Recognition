# BirdSound-Recognition

BirdSound-Recognition 是一个面向 BirdCLEF2026 数据集的鸟类声音识别课程项目。项目包含传统机器学习实验、深度学习模型、指标评估、实验报告、答辩材料，以及一个基于 FastAPI 的本地网页演示系统。

网页演示支持上传鸟类音频，自动生成波形图、Log-Mel 频谱图、MFCC 特征图，并返回 Top-k 鸟类标签预测结果。仓库中已经保留训练好的模型权重、实验表格、可视化图表和课程报告，便于复现实验展示。

## 项目特点

- 使用 BirdCLEF2026 鸟类声音数据组织实验流程。
- 包含传统机器学习模型对比：Linear SVM、KNN、ExtraTrees、Logistic Regression 等。
- 包含深度学习模型对比：CNN baseline 与 Attention-BiGRU。
- 支持 LRAP、micro F1、macro F1、Top-3 Hit Rate 等多标签识别指标。
- 提供 FastAPI + 原生 HTML/JavaScript 网页演示界面。
- 保留课程报告 PDF、LaTeX 源文件、答辩提纲和实验截图。

## 目录结构

```text
.
├── 代码/
│   ├── scripts/                  # 实验运行、报告图表生成、远程结果拉取脚本
│   ├── src/
│   │   ├── birdclef_lsw/          # 核心实验、指标、模型和推理代码
│   │   ├── birdclef_lsw_web/      # FastAPI 网页演示入口
│   │   └── birdclef_step4/        # 本地演示所需的 Step4 兼容模块
│   └── tests/                    # 指标和看板数据测试
├── 数据集/
│   └── 关键统计/                 # 报告和答辩所需的小型统计文件
├── 报告/
│   ├── 实验产物/                 # 模型权重、实验 CSV、网页截图、可视化图表
│   └── 课程报告/                 # LaTeX 源文件和最终 PDF
├── ppt/                          # 答辩提纲与 PDF
├── requirements.txt
└── README.md
```

## 环境要求

推荐环境：

- Windows 10/11、Linux 或 macOS
- Python 3.10 到 3.12
- pip

如果只运行网页演示，CPU 环境即可。重新训练深度学习模型时建议使用 NVIDIA GPU 和可用的完整 BirdCLEF2026 原始数据。

## 安装依赖

建议先创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 PowerShell 禁止激活脚本，可以临时执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 启动网页演示

在项目根目录执行：

```powershell
$env:PYTHONPATH=".\代码\src"
New-Item -ItemType Directory -Force -Path "..\web_test_audio"
python -m uvicorn birdclef_lsw_web.app:app --host 127.0.0.1 --port 7861
```

浏览器打开：

```text
http://127.0.0.1:7861
```

网页启动后，可以上传 `.wav`、`.mp3` 等音频文件进行识别。服务停止方式是在终端按 `Ctrl + C`。

## 快速自检

运行网页数据自检：

```powershell
$env:PYTHONPATH=".\代码\src"
python .\代码\scripts\run_lsw_web_smoke.py
```

正常情况下会输出类似：

```json
{
  "owner": "BirdCLEF2026 课程实验",
  "dataset": "BirdCLEF2026",
  "status": "classical_completed"
}
```

运行测试：

```powershell
$env:PYTHONPATH=".\代码\src"
python -m pytest .\代码\tests
```

## API 接口

网页服务启动后提供以下接口：

- `GET /`：网页首页。
- `GET /api/dashboard`：返回实验看板数据。
- `POST /api/predict`：上传音频文件并返回识别结果。
- `GET /api/predict-sample?name=xxx.wav`：识别 `web_test_audio` 中的示例音频。

## 实验脚本

传统机器学习实验：

```powershell
$env:PYTHONPATH=".\代码\src"
python .\代码\scripts\run_lsw_classical_experiments.py
```

深度学习实验：

```powershell
$env:PYTHONPATH=".\代码\src"
python .\代码\scripts\run_lsw_deep_experiments.py
```

完整流水线脚本：

```powershell
$env:PYTHONPATH=".\代码\src"
python .\代码\scripts\run_lsw_all.py
```

注意：完整训练依赖原始 BirdCLEF2026 音频和上游 Step2/Step4 数据产物。仓库中主要保留了展示、报告和已生成实验产物，不包含完整原始音频数据。

## 数据说明

`数据集/关键统计/` 中保留了报告和答辩需要的小型统计文件。完整 BirdCLEF2026 原始音频体积较大，不适合直接提交到 GitHub。

如需连接远程数据服务器，请在本地配置以下环境变量：

```powershell
$env:LSW_REMOTE_HOST="your.remote.host"
$env:LSW_REMOTE_PORT="22"
$env:LSW_REMOTE_USER="your_user"
$env:LSW_REMOTE_PASSWORD="your_password"
$env:LSW_REMOTE_ROOT="/home/user/lsw"
```

然后运行：

```powershell
$env:PYTHONPATH=".\代码\src"
python .\代码\scripts\pull_lsw_remote_results.py
```

仓库不会保存服务器地址、账号或密码。

## 报告与答辩材料

- 最终课程报告：`报告/课程报告/main.pdf`
- 报告 LaTeX 源文件：`报告/课程报告/main.tex`
- 实验截图与图表：`报告/实验产物/`
- 答辩提纲：`ppt/lsw_presentation_outline.md`
- 答辩提纲 PDF：`ppt/lsw_presentation_outline.pdf`

重新编译报告时，在 `报告/课程报告/` 目录执行：

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## 常见问题

### 1. `ModuleNotFoundError: No module named 'birdclef_lsw_web'`

请确认已经在项目根目录设置：

```powershell
$env:PYTHONPATH=".\代码\src"
```

### 2. `ModuleNotFoundError: No module named 'fastapi'`

请安装依赖：

```powershell
python -m pip install -r requirements.txt
```

### 3. 网页可以打开，但没有示例音频

示例音频目录默认在项目上一级的 `web_test_audio/`。可以手动创建：

```powershell
New-Item -ItemType Directory -Force -Path "..\web_test_audio"
```

也可以直接在网页中上传本地音频文件。

### 4. 是否可以直接重新训练所有模型？

仅凭本仓库不能完整重新训练所有模型，因为完整 BirdCLEF2026 原始音频没有入库。仓库中的模型权重、实验表格和报告图表已经足够运行网页演示和复现实验展示。

## 安全说明

仓库不保存远程服务器明文密码、私钥或 API token。需要远程同步时，请使用环境变量或本机 SSH 配置。
