import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import Literal, Dict, Tuple
from process import run_simulation
import numpy as np

st.set_page_config(page_title="教室音量元胞自动机", layout="wide")

st.title("🔇 教室音量演化模拟器")
st.markdown("基于元胞自动机模拟学生在不同社交压力下的音量变化。")

st.sidebar.header("⚙️ 模拟参数设置")

with st.sidebar.form("simulation_params"):
    st.subheader("1. 基础设置")
    col1, col2 = st.columns(2)
    row_num = col1.number_input("行数 (Rows)", min_value=3, max_value=20, value=6)
    col_num = col2.number_input("列数 (Cols)", min_value=3, max_value=20, value=6)
    time_steps = st.number_input(
        "模拟步数 (Time Steps)", min_value=10, max_value=500, value=50
    )
    seed = st.number_input("随机种子 (Seed)", min_value=0, value=42)

    st.subheader("2. 状态转移概率")
    p_on = st.slider(r"$P_{\text{on}}$ (保持活跃概率)", 0.0, 1.0, 0.3)
    p_off = st.slider(r"$P_{\text{off}}$ (保持静默概率)", 0.0, 1.0, 0.1)

    st.subheader("3. 音量模型系数")
    alpha = st.slider(r"$\alpha$ (模仿强度)", 0.0, 1.0, 0.5)
    beta = st.slider(r"$\beta$ (自主驱动)", 0.0, 1.0, 0.5)
    gamma = st.slider(r"$\gamma$ (突降压缩)", 0.0, 1.0, 0.5)
    lambda_rate = st.slider(r"$\lambda$ (恢复速度)", 0.0, 1.0, 0.1)

    st.subheader("4. 阈值设定")
    theta_0 = st.slider(r"$\theta$ (环境突降阈值)", 0.0, 1.0, 0.2)
    epsilon = st.slider(r"$\epsilon$ (静音阈值)", 0.0, 0.5, 0.1)
    submitted = st.form_submit_button("🚀 开始模拟")



# --- 绘图函数 ---
def plot_frame(current_data, sim_params):
    fig, ax = plt.subplots(figsize=(6, 6))
    RECTANGLE_WIDTH = 0.9
    BLANK_WIDTH = 1 - RECTANGLE_WIDTH
    rows = sim_params["row_num"]
    cols = sim_params["col_num"]
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for _, row in current_data.iterrows():
        coord = row["coord"]
        actual_vol = row["actual_volume"]
        theta_0 = sim_params["theta_0"]
        color = "#e76f51" if actual_vol > theta_0 else "#00b4d8"
        x = coord[1] + BLANK_WIDTH / 2
        y = (rows - 1 - coord[0]) + BLANK_WIDTH / 2
        rect = plt.Rectangle(
            (x, y), RECTANGLE_WIDTH, RECTANGLE_WIDTH, color=color
        )
        ax.add_patch(rect)
    
    plt.close(fig)
    return fig


# --- Streamlit 主逻辑 ---
if submitted:
    st.session_state["params"] = {
        "row_num": row_num,
        "col_num": col_num,
        "time_steps": time_steps,
        "seed": seed,
        "p_on": p_on,
        "p_off": p_off,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "lambda_rate": lambda_rate,
        "theta_0": theta_0,
        "epsilon": epsilon,
    }
    with st.spinner("正在运行模拟运算..."):
        st.session_state["df_result"] = run_simulation(st.session_state["params"])
    st.success(f"模拟完成！共生成 {len(st.session_state['df_result'])} 条状态记录。")


if "df_result" in st.session_state:
    df = st.session_state["df_result"]
    sim_params = st.session_state["params"]

    st.divider()
    col_control, col_display = st.columns(2)

    with col_control:
        st.subheader("🎥 播放控制")
        speed = st.slider("播放速度 (帧间隔秒数)", 0.01, 1.0, 0.1)
        start_btn = st.button("▶️ 自动播放所有帧")
        st.write("##")
        st.markdown("---")
        st.write("###")
        st.subheader("⏱️ 手动查看")
        manual_step = st.slider("手动选择时间步", 0, sim_params["time_steps"] - 1, 0)
        stats_placeholder = st.empty()

    with col_display:
        st.subheader("📊 教室状态热力图")
        chart_placeholder = st.empty()

    t = manual_step
    current_data = df[df["time_step"] == t]

    chart_placeholder.pyplot(plot_frame(current_data, sim_params))

    active_count = current_data["status"].sum()
    avg_volume = current_data["actual_volume"].mean()
    stats_placeholder.markdown(
        f"**当前时间步:** {t}  \n**活跃人数:** {active_count}  \n**平均音量:** {avg_volume:.3f}"
    )

    # 自动播放覆盖渲染
    if start_btn:
        for t in range(sim_params["time_steps"]):
            current_data = df[df["time_step"] == t]
            chart_placeholder.pyplot(plot_frame(current_data, sim_params))
            active_count = current_data["status"].sum()
            avg_volume = current_data["actual_volume"].mean()
            stats_placeholder.markdown(
                f"**当前时间步:** {t}  \n**活跃人数:** {active_count}  \n**平均音量:** {avg_volume:.3f}"
            )
            time.sleep(speed)

    st.divider()
    st.subheader("📈 全局趋势分析")
    stats_df = (
        df.groupby("time_step")
        .agg(avg_volume=("actual_volume", "mean"), active_ratio=("status", "mean"))
        .reset_index()
    )
    c1, c2 = st.columns(2)
    c1.line_chart(stats_df, x="time_step", y="avg_volume")
    c2.line_chart(stats_df, x="time_step", y="active_ratio")

else:
    st.info("👈 请在左侧调整参数并点击 '开始模拟' 按钮运行程序。")
