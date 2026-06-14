# 代码说明

本目录存放实验和网页相关代码。

## 目录结构

```text
src/      核心实验、模型、指标、推理和 FastAPI 网页代码
scripts/ 运行实验、拉取远程结果、生成报告图表的脚本
tests/   指标与看板数据的基础测试
```

## 本地运行方式

在项目根目录下运行时，可以使用：

```powershell
$env:PYTHONPATH=".\代码\src"
python .\代码\scripts\run_lsw_web_smoke.py
```

启动本地网页示例：

```powershell
$env:PYTHONPATH=".\代码\src"
python -m uvicorn birdclef_lsw_web.app:app --host 127.0.0.1 --port 7861
```

实验产物默认读取或写入：

```text
报告/实验产物/
```

报告源码和图表读取：

```text
报告/课程报告/
```
