import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import Literal, Dict, Tuple

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


# --- Student 数据类 ---
@dataclass
class Student:
    coord: tuple
    status: Literal[0, 1]
    sensitivity: float
    target_volume: float = 0.0
    target_ref_volume: float = 2.0

    prev_neighbor_avg_volume: float = 0.0
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.5
    lambda_rate: float = 0.1
    epsilon: float = 0.1
    theta_0: float = 1.0

    @property
    def actual_volume(self) -> float:
        return self.status * self.target_volume

    @property
    def theta(self) -> float:
        return self.sensitivity * self.theta_0


# --- 核心逻辑函数 ---
def on_off_model(student: Student, p_on: float, p_off: float) -> Literal[0, 1]:
    rand_num = np.random.rand()
    if student.status == 0:
        return 0 if rand_num < p_off else 1
    elif student.status == 1:
        return 1 if rand_num < p_on else 0
    return 1 - student.status


def get_neighbors_volume(
    student: Student, students_map: Dict[Tuple[int, int], Student]
) -> float:
    neighbor_coords = [
        (student.coord[0] - 1, student.coord[1] - 1),
        (student.coord[0] - 1, student.coord[1]),
        (student.coord[0] - 1, student.coord[1] + 1),
        (student.coord[0], student.coord[1] - 1),
        (student.coord[0], student.coord[1] + 1),
        (student.coord[0] + 1, student.coord[1] - 1),
        (student.coord[0] + 1, student.coord[1]),
        (student.coord[0] + 1, student.coord[1] + 1),
    ]
    total_volume = 0.0
    count = 0
    for coord in neighbor_coords:
        if coord in students_map:
            total_volume += students_map[coord].actual_volume
            count += 1
    return total_volume / count if count > 0 else 0.0


def update_student_volume(
    student: Student, current_neighbor_avg_volume: float
) -> float:
    if student.status == 0:
        return 0.0
    prev_neighbor_avg_volume = student.prev_neighbor_avg_volume
    delta_e = current_neighbor_avg_volume - prev_neighbor_avg_volume
    current_volume = student.target_volume
    if delta_e >= -student.theta and current_volume > student.epsilon:
        return (
            current_volume
            + student.alpha * (current_neighbor_avg_volume - current_volume)
            + student.beta * (student.target_ref_volume - current_volume)
        )
    elif delta_e <= -student.theta:
        return student.gamma * current_volume
    elif delta_e >= -student.theta and current_volume <= student.epsilon:
        return current_volume + student.lambda_rate * (
            student.target_ref_volume - current_volume
        )
    return student.target_volume


def update_student_state(student, students_map, p_on, p_off):
    new_status = on_off_model(student, p_on, p_off)
    current_neighbor_avg_volume = get_neighbors_volume(student, students_map)
    temp_student = student
    temp_student.status = new_status
    new_volume = update_student_volume(temp_student, current_neighbor_avg_volume)
    temp_student.status = student.status
    return {
        "new_status": new_status,
        "new_target_volume": new_volume,
        "new_prev_neighbor_avg_volume": current_neighbor_avg_volume,
    }


def run_simulation(params: dict) -> pd.DataFrame:
    if params.get("seed") is not None:
        np.random.seed(params["seed"])
    row_num = params["row_num"]
    col_num = params["col_num"]

    students_map = {}
    for i in range(row_num):
        for j in range(col_num):
            coord = (i, j)
            students_map[coord] = Student(
                coord=coord,
                status=1,
                sensitivity=1.0,
                alpha=params["alpha"],
                beta=params["beta"],
                gamma=params["gamma"],
                lambda_rate=params["lambda_rate"],
                epsilon=params["epsilon"],
                theta_0=params["theta_0"],
                target_ref_volume=2.0,
                target_volume=1.0,
                prev_neighbor_avg_volume=0.0,
            )

    all_history = []
    for t in range(params["time_steps"]):
        updates = {}
        for coord, student in students_map.items():
            updates[coord] = update_student_state(
                student, students_map, params["p_on"], params["p_off"]
            )
        for coord, info in updates.items():
            student = students_map[coord]
            student.status = info["new_status"]
            student.target_volume = info["new_target_volume"]
            student.prev_neighbor_avg_volume = info["new_prev_neighbor_avg_volume"]
        for student in students_map.values():
            all_history.append(
                {
                    "time_step": t,
                    "coord": student.coord,
                    "status": student.status,
                    "target_volume": student.target_volume,
                    "actual_volume": student.actual_volume,
                }
            )
    return pd.DataFrame(all_history)


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
