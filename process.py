# process.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Dict, Tuple, List


# --- 数据类定义 ---
@dataclass
class Student:
    coord: tuple
    status: Literal[0, 1]
    sensitivity: float

    # 模型参数字段
    alpha: float
    beta: float
    gamma: float
    lambda_rate: float
    theta: float
    epsilon: float
    target_volume: float
    prev_neighbor_avg_volume: float = 0.0

    @property
    def ref_volume(self) -> float:
        return self.status * self.target_volume


# --- 辅助逻辑函数 ---


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
            total_volume += students_map[coord].ref_volume
            count += 1
    return total_volume / count if count > 0 else 0.0


def update_student_volume(
    student: Student, current_neighbor_avg_volume: float
) -> float:
    if student.status == 0:
        return 0.0

    prev_neighbor_avg = student.prev_neighbor_avg_volume
    delta_e = current_neighbor_avg_volume - prev_neighbor_avg
    current_vol = student.target_volume

    # 规则应用
    if current_vol <= student.epsilon:
        # 恢复阶段
        return current_vol + student.lambda_rate * (student.target_volume - current_vol)
    elif delta_e < -student.theta and current_vol > student.epsilon:
        # 突然安静
        return student.gamma * current_vol
    elif delta_e >= -student.theta and current_vol > student.epsilon:
        # 常规调节
        imitation = student.alpha * (student.target_volume - current_vol)
        self_drive = student.beta * (student.target_volume - current_vol)
        return current_vol + imitation + self_drive

    return current_vol


def update_single_step(student: Student, students_map: dict, p_on: float, p_off: float):
    """计算单个学生的下一步状态，返回更新字典"""
    # 1. 计算新状态
    new_status = on_off_model(student, p_on, p_off)

    # 2. 获取当前邻居音量 (这将成为下一刻的历史值)
    current_neighbor_avg = get_neighbors_volume(student, students_map)

    # 3. 计算新音量 (使用假想的新状态进行计算)
    temp_student = student
    original_status = student.status
    temp_student.status = new_status  # 临时修改以计算体积

    new_volume = update_student_volume(temp_student, current_neighbor_avg)

    temp_student.status = original_status  # 恢复

    return {
        "new_status": new_status,
        "new_target_volume": new_volume,
        "new_prev_neighbor_avg_volume": current_neighbor_avg,
    }


# --- 主模拟函数 ---


def run_simulation(params: dict) -> pd.DataFrame:
    """
    接收参数字典，运行模拟，返回 DataFrame
    params 包含: row_num, col_num, time_steps, p_on, p_off,
                 alpha, beta, gamma, lambda, theta, epsilon, seed
    """
    # 设置随机种子
    if params.get("seed") is not None:
        np.random.seed(params["seed"])

    row_num = params["row_num"]
    col_num = params["col_num"]

    # 1. 初始化学生
    students_map = {}
    for i in range(row_num):
        for j in range(col_num):
            coord = (i, j)
            students_map[coord] = Student(
                coord=coord,
                status=1,  # 默认初始活跃
                sensitivity=1.0,
                alpha=params["alpha"],
                beta=params["beta"],
                gamma=params["gamma"],
                lambda_rate=params["lambda_rate"],
                theta=params["theta"],
                epsilon=params["epsilon"],
                target_volume=1.0,  # 默认目标音量
                prev_neighbor_avg_volume=0.0,
            )

    all_history = []

    # 2. 时间步循环
    for t in range(params["time_steps"]):
        # --- 阶段 1: 计算所有更新 ---
        updates = {}
        for coord, student in students_map.items():
            updates[coord] = update_single_step(
                student, students_map, params["p_on"], params["p_off"]
            )

        # --- 阶段 2: 应用更新 ---
        for coord, info in updates.items():
            student = students_map[coord]
            student.status = info["new_status"]
            student.target_volume = info["new_target_volume"]
            student.prev_neighbor_avg_volume = info["new_prev_neighbor_avg_volume"]

        # --- 阶段 3: 记录数据 ---
        for student in students_map.values():
            all_history.append(
                {
                    "time_step": t,
                    "coord": student.coord,
                    "status": student.status,
                    "target_volume": student.target_volume,
                    "ref_volume": student.ref_volume,
                }
            )

    return pd.DataFrame(all_history)
