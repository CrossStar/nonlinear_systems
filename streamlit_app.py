import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import time
import io
import tempfile
import os

# 1. 页面配置
st.set_page_config(page_title="基于元胞自动机的教室声音模拟系统", layout="wide")

# -------------------------------------------------------
# 核心数学函数
# -------------------------------------------------------
def interaction_factor(x, y):
    return math.log1p(x)/math.log(2) * 0.5 * (1 + math.tanh(5*(y+1.5)))

def random_signal_a(t):
    return 1 + math.tanh(7 * (
        math.sin(math.sqrt(3)*0.7*t) +
        math.sin(math.sqrt(5)*0.7*t) +
        0.332*math.sin(math.sqrt(16)*0.7*t) +
        math.sin(math.sqrt(14)*0.7*t) +
        1.02*math.sin(math.sqrt(2.5803)*0.7*t)
    ))

def random_signal_b(t):
    return np.mean([random_signal_a(t+i*10) for i in range(10)])

def random_signal_g(t):
    return np.mean([random_signal_b(t+i*100) for i in range(4)])

# -------------------------------------------------------
# 模拟计算逻辑 (缓存)
# -------------------------------------------------------
@st.cache_data
def run_simulation(time_step, time_start, time_end, row_num, col_num):
    F_history = {round(t, 5): random_signal_g(t) 
                 for t in np.arange(time_start, 0+time_step, time_step)}
    
    time_points = []
    volume_values = []
    F_record_dict = {}
    
    steps = np.arange(0, time_end + time_step, time_step)
    
    progress_text = "正在进行数学模拟..."
    my_bar = st.progress(0, text=progress_text)
    total_steps = len(steps)

    for idx, t in enumerate(steps):
        if idx % 50 == 0:
            my_bar.progress(min(idx / total_steps, 1.0), text=f"模拟中: {t:.1f}/{time_end}s")

        t_r = lambda x: round(x, 5)
        current_t = t_r(t)

        keys_mean = [t_r(t - i * time_step) for i in range(1, 7)]
        vals_mean = [F_history.get(k, 0) for k in keys_mean]
        recent_mean = np.mean(vals_mean)
        
        keys_diff_1 = [t_r(t - i * time_step) for i in range(1, 6)]
        keys_diff_2 = [t_r(t - i * time_step - 0.5) for i in range(1, 6)]
        vals_diff_1 = [F_history.get(k, 0) for k in keys_diff_1]
        vals_diff_2 = [F_history.get(k, 0) for k in keys_diff_2]
        recent_diff = 2 * (np.mean(vals_diff_1) - np.mean(vals_diff_2))
        
        sig_g = random_signal_g(t)
        new_val = sig_g * interaction_factor(recent_mean, recent_diff) + 0.001 * sig_g
        F_history[current_t] = new_val
        F_record_dict[current_t] = new_val
        
        del_key = t_r(t - 0.8)
        if del_key in F_history:
            del F_history[del_key]
        
        time_points.append(current_t)
        volume_values.append(new_val)
        
    my_bar.empty()
    return time_points, volume_values, F_record_dict

# -------------------------------------------------------
# 辅助函数：生成网格
# -------------------------------------------------------
def get_grid_data(f_val, r, c, seed):
    np.random.seed(seed)
    random_matrix = np.random.rand(r, c)
    return f_val * (0.9 + 0.2 * random_matrix)

# -------------------------------------------------------
# 辅助函数：生成 GIF (修复版)
# -------------------------------------------------------
def generate_gif(time_points, f_dict, row_num, col_num, v_min, v_max, start_t, end_t, fps=10):
    """
    生成 GIF 并返回 BytesIO 对象 (使用临时文件中转以修复路径报错)
    """
    # 筛选时间段
    valid_indices = [i for i, t in enumerate(time_points) if start_t <= t <= end_t]
    
    # 智能降采样：如果帧数过多，自动抽帧以防止生成过慢
    step = 1
    if len(valid_indices) > 200:
        step = len(valid_indices) // 100
    
    indices_to_plot = valid_indices[::step]
    
    if not indices_to_plot:
        return None

    # 创建绘图对象
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # 初始化第一帧
    first_idx = indices_to_plot[0]
    initial_grid = get_grid_data(f_dict[time_points[first_idx]], row_num, col_num, seed=first_idx)
    im = ax.imshow(initial_grid, vmin=v_min, vmax=v_max, cmap='Blues', interpolation='nearest')
    title = ax.set_title(f"Time: {time_points[first_idx]:.2f} s")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    # 更新函数
    def update(frame_idx):
        t_curr = time_points[frame_idx]
        f_val = f_dict[t_curr]
        grid = get_grid_data(f_val, row_num, col_num, seed=frame_idx)
        im.set_data(grid)
        title.set_text(f"Time: {t_curr:.2f} s")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=indices_to_plot, blit=False)
    
    buf = None
    tmp_filename = None
    
    try:
        # 1. 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        # 2. 将动画保存到这个真实的临时路径
        ani.save(tmp_filename, writer='pillow', fps=fps)
        
        # 3. 重新以二进制读取文件内容到内存
        with open(tmp_filename, "rb") as f:
            buf = io.BytesIO(f.read())
            
    except Exception as e:
        st.error(f"GIF 生成出错: {e}")
    finally:
        # 4. 清理：删除临时文件
        plt.close(fig) # 关闭 matplotlib 图形释放内存
        if tmp_filename and os.path.exists(tmp_filename):
            try:
                os.remove(tmp_filename)
            except Exception:
                pass 

    if buf:
        buf.seek(0)
    return buf

# -------------------------------------------------------
# 界面布局
# -------------------------------------------------------

# === Sidebar: 参数配置 ===
with st.sidebar:
    st.header("⚙️ 参数配置")
    st.markdown("### 1. 时间设置")
    time_end = st.slider("模拟时长 (Time End)", 20.0, 1000.0, 60.0, step=10.0)
    time_step = st.number_input("时间步长 (Step)", value=0.05, format="%.2f", disabled=True)
    
    st.markdown("### 2. 网格设置")
    c1, c2 = st.columns(2)
    with c1: row_num = st.number_input("行数", 5, 50, 10)
    with c2: col_num = st.number_input("列数", 5, 50, 10)

    st.markdown("---")
    start_btn = st.button("▶ 开始模拟", type="primary", use_container_width=True)

# 初始化 Session State
if 'sim_result' not in st.session_state:
    st.session_state['sim_result'] = None
if 'gif_buffer' not in st.session_state:
    st.session_state['gif_buffer'] = None

# 执行模拟
if start_btn:
    st.session_state['gif_buffer'] = None # 重置旧的GIF
    t_pts, vols, f_dict = run_simulation(time_step, -1, time_end, row_num, col_num)
    st.session_state['sim_result'] = {
        'time': t_pts,
        'volume': vols,
        'f_dict': f_dict,
        'v_min': min(vols) * 0.9,
        'v_max': max(vols) * 1.1
    }

# === Main Layout ===
st.title("基于元胞自动机的教室声音模拟系统")

col_left, col_right = st.columns([1, 1.2], gap="large")

# --- 左栏：数学公式 ---
with col_left:
    st.subheader("1. 相关数学模型")
    
    st.markdown("**1. 基础随机波 $a(t)$:**")
    st.latex(r"""
    a(t) = 1 + \tanh\left[ 7 \cdot \left( 
    \begin{aligned}
    &\sin(0.7\sqrt{3}t) + \sin(0.7\sqrt{5}t) + \\
    &0.332\sin(2.8t) + \sin(0.7\sqrt{14}t) + \\
    &1.02\sin(0.7\sqrt{2.5803}t)
    \end{aligned}
    \right) \right]
    """)
    with st.expander("👁️ 查看 a(t) 波形", expanded=True):
        t_preview = np.linspace(0, 100, 200)
        y_preview = [random_signal_a(t) for t in t_preview]
        st.line_chart(pd.DataFrame({"Time": t_preview, "a(t)": y_preview}).set_index("Time"), height=120, color="#FF4B4B")

    st.markdown("---")
    st.markdown("**2. 相互作用迭代 $F(t)$:**")
    st.latex(r"I(x, y) = \frac{\ln(1+x)}{\ln(2)} \cdot 0.5 \cdot (1 + \tanh(5(y+1.5)))")
    st.latex(r"F(t) = G(t) \cdot I(\mu_{recent}, \Delta_{recent}) + 0.001 \cdot G(t)")

# --- 右栏：热力图与GIF导出 ---
with col_right:
    st.subheader("2. 动态热力图 (Heatmap)")
    
    if st.session_state['sim_result'] is None:
        st.info("👈 请先在左侧点击“开始模拟”")
        st.markdown("<br>"*5, unsafe_allow_html=True)
    else:
        data = st.session_state['sim_result']
        time_points = data['time']
        f_dict = data['f_dict']
        
        # 播放控制
        c_ctrl_1, c_ctrl_2 = st.columns([1, 2])
        with c_ctrl_1:
            auto_play = st.toggle("🔄 自动播放", value=False)
        with c_ctrl_2:
            if not auto_play:
                frame_idx = st.slider("预览时间轴", 0, len(time_points)-1, 0, label_visibility="collapsed")
            else:
                st.caption("正在播放动画...")

        # 绘图区域
        heatmap_placeholder = st.empty()
        
        def plot_frame(idx):
            t_curr = time_points[idx]
            f_val = f_dict[t_curr]
            grid = get_grid_data(f_val, row_num, col_num, seed=idx)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(grid, vmin=data['v_min'], vmax=data['v_max'], 
                           cmap='Blues', interpolation='nearest')
            ax.set_title(f"Time: {t_curr:.2f} s | Volume: {f_val:.3f}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            return fig

        if auto_play:
            for i in range(0, len(time_points), 2):
                fig = plot_frame(i)
                heatmap_placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(0.05)
        else:
            fig = plot_frame(frame_idx)
            heatmap_placeholder.pyplot(fig)
            plt.close(fig)
        
        # ---------------------------------------------------
        # GIF 导出区域 (已修改为可折叠)
        # ---------------------------------------------------
        st.markdown("---")
        # 使用 st.expander 替代 st.subheader + st.container
        with st.expander("📤 导出 GIF 动画 (点击展开)", expanded=False):
            st.markdown("##### 1. 选择时间范围")
            # 双向滑块选择时间段
            gif_range = st.slider(
                "GIF 截取时段 (秒)",
                min_value=float(time_points[0]),
                max_value=float(time_points[-1]),
                value=(0.0, min(10.0, float(time_points[-1]))), # 默认前10秒
                step=1.0
            )
            
            st.markdown("##### 2. 生成与下载")
            col_in, col_btn = st.columns([2, 1])
            with col_in:
                # 允许用户自定义文件名
                custom_filename = st.text_input("文件名 (无需后缀)", value="simulation_result")
                fps_val = st.number_input("帧率 (FPS)", 5, 30, 10)
            
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True) # 布局对齐
                generate_gif_btn = st.button("生成 GIF", icon="🎬", use_container_width=True)
            
            # 生成逻辑
            if generate_gif_btn:
                with st.spinner("正在渲染 GIF (可能需要几秒钟)..."):
                    gif_buffer = generate_gif(
                        time_points, f_dict, row_num, col_num, 
                        data['v_min'], data['v_max'], 
                        start_t=gif_range[0], end_t=gif_range[1], fps=fps_val
                    )
                    st.session_state['gif_buffer'] = gif_buffer
                
                if st.session_state['gif_buffer']:
                    st.success("✅ 生成成功！请点击下方按钮保存。")
                else:
                    st.warning("⚠️ 所选时间段内没有数据，请调整范围。")
            
            # 下载按钮
            if st.session_state['gif_buffer'] is not None:
                final_filename = f"{custom_filename}.gif"
                st.download_button(
                    label=f"⬇️ 下载 {final_filename}",
                    data=st.session_state['gif_buffer'],
                    file_name=final_filename,
                    mime="image/gif",
                    type="primary",
                    use_container_width=True
                )

# --- 底部：趋势图 ---
st.markdown("---")
st.subheader("3. 总体音量变化")

if st.session_state['sim_result'] is not None:
    data = st.session_state['sim_result']
    chart_df = pd.DataFrame({
        "Time (s)": data['time'],
        "Volume": data['volume']
    }).set_index("Time (s)")
    
    # 间隔 0.5s 取样显示
    chart_df = chart_df.iloc[::int(0.5 / time_step), :]
    st.line_chart(chart_df, height=350, width='stretch')