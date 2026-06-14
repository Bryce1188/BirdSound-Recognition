# 报告说明

本目录包含课程报告正文和支撑实验产物。

## 目录结构

```text
课程报告/  LaTeX 源文件、最终 PDF、报告图表和自动生成表格
实验产物/  实验 CSV、模型权重、网页截图、图表和网页缓存图
```

最终报告：

```text
课程报告/main.pdf
```

主要源码：

```text
课程报告/main.tex
```

重新编译报告时，在 `课程报告/` 目录运行：

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```
