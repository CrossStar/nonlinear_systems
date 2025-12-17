import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
import tempfile
import os

# =================== 1. 核心逻辑函数 ===================


def get_neighborhood_avg(v, i, j, R, N, M):
    """获取以 (i,j) 为中心，半径为 R 的邻域的平均音量"""
    i_min, i_max = max(0, i - R), min(N, i + R + 1)
    j_min, j_max = max(0, j - R), min(M, j + R + 1)
    region = v[i_min:i_max, j_min:j_max]
    return np.mean(region)


def update_step(
    X,
    g,
    g_ref,
    v,
    last_v,
    theta,
    N,
    M,
    R,
    p_on,
    p_off,
    epsilon,
    alpha,
    beta,
    gamma,
    lambda_recover,
):
    """执行一步模拟更新"""
    next_g = g.copy()
    next_v = np.zeros((N, M))
    next_X = X.copy()

    for i in range(N):
        for j in range(M):
            v_avg = get_neighborhood_avg(v, i, j, R, N, M)
            last_v_avg = get_neighborhood_avg(last_v, i, j, R, N, M)
            delta_v_avg = v_avg - last_v_avg

            # 更新说话状态
            if X[i, j] == 1:
                if np.random.rand() > p_on:
                    next_X[i, j] = 0
            else:
                if np.random.rand() > p_off:
                    next_X[i, j] = 1

            # 音量更新逻辑
            curr_g = g[i, j]
            curr_theta = theta[i, j]

            if delta_v_avg >= -curr_theta and curr_g > epsilon:
                next_g[i, j] = (
                    curr_g + alpha * (v_avg - curr_g) + beta * (g_ref[i, j] - curr_g)
                )
            elif delta_v_avg < -curr_theta:
                next_g[i, j] = gamma * curr_g
            elif delta_v_avg >= -curr_theta and curr_g <= epsilon:
                next_g[i, j] = curr_g + lambda_recover * (g_ref[i, j] - curr_g)

            next_g[i, j] = np.clip(next_g[i, j], 0, 1)
            next_v[i, j] = next_X[i, j] * next_g[i, j]

    return next_X, next_g, next_v


# =================== 2. 页面布局 ===================

st.set_page_config(page_title="教室音量场生成器", layout="wide")

st.title("🔇 教室音量场")

# --- 侧边栏：所有参数 ---
with st.sidebar:
    st.header("⚙️ 参数设置")

    # 1. 模拟时长设置 (关键修改)
    st.subheader("1. 生成设置")
    total_steps = st.number_input(
        "模拟总步数 (T)", value=200, min_value=10, max_value=2000, step=10
    )
    fps_val = st.slider("GIF 帧率 (FPS)", min_value=5, max_value=60, value=20)

    st.divider()

    # 2. 环境参数
    st.subheader("2. 环境参数")
    N = st.number_input("行数 (N)", value=8, min_value=3)
    M = st.number_input("列数 (M)", value=8, min_value=3)
    R = st.slider("邻域半径 (R)", 1, 5, 3)

    # 3. 行为参数
    with st.expander("高级行为参数 (点击展开)"):
        alpha = st.slider("模仿强度 (Alpha)", 0.0, 0.5, 0.02)
        beta = st.slider("自主调节 (Beta)", 0.0, 0.5, 0.03)
        gamma = st.slider("惊吓压缩 (Gamma)", 0.0, 1.0, 0.0)
        lambda_recover = st.slider("恢复速度", 0.0, 0.1, 0.005, format="%.3f")
        p_on = st.slider("P_on (保持说话)", 0.5, 1.0, 0.95)
        p_off = st.slider("P_off (保持闭嘴)", 0.5, 1.0, 0.70)
        theta_0 = st.number_input("安静阈值基准", value=0.1)
        epsilon = st.number_input("静音阈值", value=0.1)

    # 生成按钮
    generate_btn = st.button(
        "🚀 开始生成模拟", type="primary", use_container_width=True
    )

# =================== 3. 主逻辑：生成过程 ===================

if generate_btn:
    # --- A. 初始化状态 ---
    np.random.seed(42)
    X = np.random.choice([0, 1], size=(N, M))
    g_ref = np.clip(np.random.normal(0.5, 0.15, (N, M)), 0.1, 1.0)
    g = np.clip(np.random.normal(0.5, 0.15, (N, M)), 0.0, 1.0)
    a = np.clip(np.random.normal(0.5, 0.15, (N, M)), 0.5, 1)
    theta = a * theta_0
    v = X * g
    last_v = v.copy()

    # 用于存储每一帧的数据
    history_v = [v.copy()]
    history_mean = [np.mean(v)]

    # --- B. 运行模拟 (纯数值计算，速度快) ---
    progress_bar = st.progress(0, text="正在进行数值模拟...")

    for t in range(total_steps):
        next_X, next_g, next_v = update_step(
            X,
            g,
            g_ref,
            v,
            last_v,
            theta,
            N,
            M,
            R,
            p_on,
            p_off,
            epsilon,
            alpha,
            beta,
            gamma,
            lambda_recover,
        )

        # 更新变量
        last_v = v.copy()
        X, g, v = next_X, next_g, next_v

        # 记录数据
        history_v.append(v.copy())
        history_mean.append(np.mean(v))

        # 更新进度条 (为了性能，每10步更新一次UI)
        if t % 10 == 0:
            progress_bar.progress(
                int((t / total_steps) * 50), text=f"正在模拟: 第 {t}/{total_steps} 步"
            )

    progress_bar.progress(50, text="模拟完成，正在渲染 GIF 动画 (这可能需要几秒钟)...")

    # --- C. 生成 GIF (Matplotlib) ---
    try:
        # 创建临时文件
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        gif_path = tfile.name

        # 绘图设置
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.cm.Blues
        norm = colors.Normalize(vmin=0, vmax=1)

        # 初始化第一帧
        im = ax.imshow(history_v[0], cmap=cmap, norm=norm, interpolation="nearest")
        plt.colorbar(im, ax=ax, label="Volume")
        ax.set_xticks(np.arange(-0.5, M, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)
        title_text = ax.set_title(f"Simulation t=0")

        # 动画更新函数
        def animate(i):
            im.set_array(history_v[i])
            title_text.set_text(f"Simulation t={i}\nMean Volume: {history_mean[i]:.3f}")
            return [im, title_text]

        # 编译动画
        ani = animation.FuncAnimation(
            fig, animate, frames=len(history_v), interval=1000 / fps_val, blit=False
        )

        # 保存 GIF (使用 Pillow writer)
        ani.save(gif_path, writer="pillow", fps=fps_val)
        plt.close(fig)

        progress_bar.progress(100, text="渲染完成！")

        # --- D. 结果展示 ---
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🎬 模拟结果动画")
            st.image(gif_path, caption=f"Total Steps: {total_steps}, FPS: {fps_val}")

            # 下载按钮
            with open(gif_path, "rb") as f:
                btn = st.download_button(
                    label="💾 下载 GIF 动画",
                    data=f,
                    file_name="classroom_simulation.gif",
                    mime="image/gif",
                )

        with col2:
            st.subheader("📈 全局平均音量趋势")
            st.line_chart(history_mean)
            st.success(f"模拟结束。最终平均音量: {history_mean[-1]:.4f}")

        # 清理临时文件 (可选，但在Windows上直接删除可能会因为占用而报错，暂留)
        # os.unlink(gif_path)

    except Exception as e:
        st.error(f"生成 GIF 时发生错误: {e}")
