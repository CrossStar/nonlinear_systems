import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import process  # 导入后端计算模块

# 设置页面配置
st.set_page_config(page_title="教室音量元胞自动机", layout="wide")

st.title("🔇 教室音量演化模拟器")
st.markdown("基于元胞自动机 (CA) 模拟学生在不同社交压力下的音量变化。")

# ==========================================
# 1. 侧边栏：参数配置
# ==========================================
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
    p_on = st.slider("P_on (保持活跃概率)", 0.0, 1.0, 0.3)
    p_off = st.slider("P_off (保持静默概率)", 0.0, 1.0, 0.1)

    st.subheader("3. 音量模型系数")
    alpha = st.slider("α (模仿强度)", 0.0, 1.0, 0.5)
    beta = st.slider("β (自主驱动)", 0.0, 1.0, 0.5)
    gamma = st.slider("γ (突降压缩)", 0.0, 1.0, 0.5)
    lambda_rate = st.slider("λ (恢复速度)", 0.0, 1.0, 0.1)

    st.subheader("4. 阈值设定")
    theta = st.slider("θ (环境突降阈值)", 0.0, 1.0, 0.2)
    epsilon = st.slider("ε (静音阈值)", 0.0, 0.5, 0.1)

    submitted = st.form_submit_button("🚀 开始模拟")

# ==========================================
# 2. 运行模拟逻辑
# ==========================================
if submitted:
    params = {
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
        "theta": theta,
        "epsilon": epsilon,
    }
    with st.spinner("正在运行模拟运算..."):
        df_result = process.run_simulation(params)

    st.session_state["df_result"] = df_result
    st.session_state["params"] = params
    st.success(f"模拟完成！共生成 {len(df_result)} 条状态记录。")


# ==========================================
# 3. 绘图封装函数 (为了复用)
# ==========================================
def plot_frame(current_data, sim_params):
    """绘制单个时间步的图像，返回 fig 对象"""
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
        status = row["status"]
        vol = row["original_volume"]

        # 颜色：红(Active) / 蓝(Quiet)
        color = "#e76f51" if status == 1 else "#00b4d8"

        # 透明度映射
        alpha_val = 0.3 + (vol * 0.7)
        alpha_val = min(max(alpha_val, 0.3), 1.0)

        x = coord[1] + BLANK_WIDTH / 2
        y = (rows - 1 - coord[0]) + BLANK_WIDTH / 2

        rect = plt.Rectangle(
            (x, y), RECTANGLE_WIDTH, RECTANGLE_WIDTH, color=color, alpha=alpha_val
        )
        ax.add_patch(rect)

    return fig


# ==========================================
# 4. 结果可视化界面
# ==========================================
if "df_result" in st.session_state:
    df = st.session_state["df_result"]
    sim_params = st.session_state["params"]

    st.divider()

    # 布局
    col_control, col_display = st.columns([1, 2])

    # --- 左侧：控制区 ---
    with col_control:
        st.subheader("🎥 播放控制")

        # 播放速度控制
        speed = st.slider("播放速度 (帧间隔秒数)", 0.01, 1.0, 0.1)

        # 播放按钮
        start_btn = st.button("▶️ 自动播放所有帧")

        st.markdown("---")
        st.subheader("⏱️ 手动查看")
        # 手动滑块 (如果正在自动播放，这个滑块不会动，但不影响程序运行)
        manual_step = st.slider(
            "手动选择时间步",
            min_value=0,
            max_value=sim_params["time_steps"] - 1,
            value=0,
        )

        # 统计数据显示
        # 这里的逻辑是：如果是点击了播放，我们在循环里更新统计；
        # 如果没播放，我们显示 manual_step 的统计。
        # 为了简单，我们在下方统一处理统计数据的占位符。
        stats_placeholder = st.empty()

    # --- 右侧：绘图区 ---
    with col_display:
        st.subheader("📊 教室状态热力图")
        # 创建一个空容器，用于动态放置图表
        chart_placeholder = st.empty()

    # ==========================================
    # 5. 渲染逻辑 (自动播放 vs 手动)
    # ==========================================

    if start_btn:
        # --- 自动播放模式 ---
        progress_bar = st.progress(0)
        total_steps = sim_params["time_steps"]

        for t in range(total_steps):
            # 1. 获取数据
            current_data = df[df["time_step"] == t]

            # 2. 绘制并放入占位符
            fig = plot_frame(current_data, sim_params)
            chart_placeholder.pyplot(fig)
            plt.close(fig)  # 重要：关闭图形释放内存

            # 3. 更新统计信息占位符
            active_count = current_data["status"].sum()
            avg_volume = current_data["ref_volume"].mean()
            stats_placeholder.markdown(
                f"""
                **当前时间步:** {t}  
                **活跃人数:** {active_count}  
                **平均音量:** {avg_volume:.3f}
                """
            )

            # 4. 更新进度条和休眠
            progress_bar.progress((t + 1) / total_steps)
            time.sleep(speed)

        st.success("播放结束")

    else:
        # --- 手动模式 (默认) ---
        t = manual_step
        current_data = df[df["time_step"] == t]

        # 1. 绘图
        fig = plot_frame(current_data, sim_params)
        chart_placeholder.pyplot(fig)  # 放入同一个占位符

        # 2. 统计
        active_count = current_data["status"].sum()
        avg_volume = current_data["ref_volume"].mean()
        stats_placeholder.markdown(
            f"""
            **当前时间步:** {t}  
            **活跃人数:** {active_count}  
            **平均音量:** {avg_volume:.3f}
            """
        )

    # --- 底部全局趋势图 ---
    st.divider()
    st.subheader("📈 全局趋势分析")
    stats_df = (
        df.groupby("time_step")
        .agg(avg_volume=("ref_volume", "mean"), active_ratio=("status", "mean"))
        .reset_index()
    )

    c1, c2 = st.columns(2)
    c1.line_chart(stats_df, x="time_step", y="avg_volume")
    c2.line_chart(stats_df, x="time_step", y="active_ratio")

else:
    st.info("👈 请在左侧调整参数并点击 '开始模拟' 按钮运行程序。")
