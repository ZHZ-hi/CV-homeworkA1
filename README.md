# 图像处理工具

这是一个用于颜色空间转换、图像插值变换和通道调节的 Streamlit 应用。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud 部署

1. 将项目上传到 GitHub。
2. 在 Streamlit Cloud 新建应用。
3. Main file path 填写 `streamlit_app.py`。
4. Python 版本建议选择 `3.11` 或 `3.12`。

部署时不需要上传 `venv/`、`uploads/`、`output_channels/`、`output_interpolation/` 等本地生成目录。
