# 翅片管式蒸发器数字孪生 | Finned-Tube Evaporator Digital Twin

基于 ASHRAE 标准的翅片管式蒸发器交互式仿真应用。

An interactive simulation app for finned-tube evaporators, built on ASHRAE standards and verified thermodynamic correlations.

## ✨ 功能特性

- **湿空气热工计算** — ASHRAE Fundamentals Ch.1 (Hyland-Wexler 方程)
- **制冷剂物性** — CoolProp (Helmholtz 状态方程) + 内置多项式后备方案
- **三层仿真架构** — 几何模型 → 物理机理 → 性能仿真
- **干/湿工况自动判别** — Threlkeld 焓法，壁面温度二分法求解
- **压降仿真** — 均匀流模型 + 加速压降 + U 型弯头局部损失
- **室内/室外侧预设** — 一键切换蒸发器工况参数
- **可视化** — 焓湿图、敏感性分析、沿程温度分布曲线

## 🧮 核心模型

| 模块 | 方法 | 参考 |
|------|------|------|
| 翅片效率 | 板式翅片等效圆形法 + Harper-Brown 修正 | ASHRAE Fundamentals |
| 管外换热 | Gray & Webb 关联式 | ASHRAE Systems & Equipment Ch.22 |
| 管内沸腾 | Shah / Kandlikar 关联式 | — |
| 湿工况 | Threlkeld 焓法 | ASHRAE Standard 33-2000 |
| 压降 | 均匀流模型 (Blasius 摩擦因子) | ASHRAE Fundamentals 2013 Ch.5 |

## 🛠 技术栈

Python 3.12+ · Streamlit · NumPy · SciPy · Plotly · CoolProp

## 🚀 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ 部署到 Streamlit Community Cloud

1. 将本仓库推送到 GitHub (公开仓库)
2. 访问 [share.streamlit.io](https://share.streamlit.io)
3. 连接 GitHub 仓库，配置：
   - **入口文件**: `app.py`
   - **Python 版本**: 3.12
   - **Secrets**: 无需配置 (本应用不依赖密钥)
4. 点击 Deploy，等待构建完成

## 📐 支持的制冷剂

R22 · R134a · R410A · R32 · R290 · R407C · R404A · R23 · R744

## 📚 参考标准

- ASHRAE Fundamentals 2013 (Ch.1 湿空气, Ch.4 传热, Ch.5 压降)
- ASHRAE Systems & Equipment 2012 (Ch.22 冷却盘管)
- ASHRAE Standard 33-2000 (强制对流盘管测试方法)
- JB/T 7659.5-95 / GB/T 23130-2008 / GB/T 47234-2026
