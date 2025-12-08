import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Dict, Tuple

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
    student: Student, students_map: Dict[Tuple[int, int], Student], R: int = 1
) -> float:
    x, y = student.coord

    total = 0.0
    count = 0

    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            # 跳过自己
            if dx == 0 and dy == 0:
                continue

            coord = (x + dx, y + dy)
            s = students_map.get(coord)
            if s is not None:
                total += s.actual_volume
                count += 1

    return total / count if count else 0.0


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


def update_student_state(student, students_map, p_on, p_off, R):
    new_status = on_off_model(student, p_on, p_off)
    current_neighbor_avg_volume = get_neighbors_volume(student, students_map, R)
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
                student, students_map, params["p_on"], params["p_off"], params["R"]
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
