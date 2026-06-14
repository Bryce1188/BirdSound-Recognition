# 数据集说明

本目录只保留本地关键统计文件，不放完整 BirdCLEF2026 原始音频。原始音频体积较大，远程服务器上已经保留完整数据，报告和代码均按 BirdCLEF2026 数据集组织实验。

## 本地关键数据

`关键统计/` 中保留了报告和答辩会用到的小文件：

- `lsw_dataset_statistics.csv`：数据规模与样本窗口统计
- `lsw_top_labels.csv`：高频标签统计
- `lsw_feature_families.csv`：声学特征家族与维度说明
- `lsw_manifest_snapshot.json`：传统机器学习实验快照
- `lsw_deep_manifest_snapshot.json`：深度学习实验快照

## 远程完整数据位置

远程主目录：

```text
/home/user/lsw
```

BirdCLEF2026 数据目录：

```text
/home/user/lsw/birdclef-2026/
```

关键子目录：

```text
/home/user/lsw/birdclef-2026/train_audio/
/home/user/lsw/birdclef-2026/train_soundscapes/
/home/user/lsw/birdclef-2026/*.csv
```

## 远程数据连接方式

完整 BirdCLEF2026 原始音频体积较大，仓库中不保存服务器地址、账号或密码。
如果需要连接远程数据服务器，请向项目维护者获取连接信息，并通过环境变量配置：

```powershell
$env:LSW_REMOTE_HOST="your.remote.host"
$env:LSW_REMOTE_PORT="22"
$env:LSW_REMOTE_USER="your_user"
$env:LSW_REMOTE_PASSWORD="your_password"
$env:LSW_REMOTE_ROOT="/home/user/lsw"
```

然后可以运行 `代码/scripts/pull_lsw_remote_results.py` 拉取远程实验产物。

## 远程运行说明

远程 Python：

```text
/usr/local/iCompute/bin/python
```

网页启动命令：

```bash
cd /home/user/lsw
/usr/local/iCompute/bin/python -m uvicorn src.birdclef_lsw_web.app:app --host 127.0.0.1 --port 7860
```

网页访问仍通过本机反向代理隧道访问远程 `127.0.0.1:7860`。
