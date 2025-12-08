import numpy as np
import pandas as pd
from dataclasses import dataclass, replace
from typing import Literal, Dict, Tuple


@dataclass
class Student:
    coord: Tuple[int, int]
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
        return float(self.status) * float(self.target_volume)

    @property
    def theta(self) -> float:
        return self.sensitivity * self.theta_0


def on_off_model(student: Student, p_on: float, p_off: float) -> Literal[0, 1]:
    r = np.random.rand()
    if student.status == 0:
        return 0 if r < p_off else 1
    if student.status == 1:
        return 1 if r < p_on else 0
    return 1 - student.status


def get_neighbors_volume(
    student: Student, students_map: Dict[Tuple[int, int], Student]
) -> float:
    x, y = student.coord
    neighbors = (
        (x - 1, y - 1),
        (x - 1, y),
        (x - 1, y + 1),
        (x, y - 1),
        (x, y + 1),
        (x + 1, y - 1),
        (x + 1, y),
        (x + 1, y + 1),
    )
    total = 0.0
    count = 0
    for coord in neighbors:
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

    prev = student.prev_neighbor_avg_volume
    delta_e = current_neighbor_avg_volume - prev
    cur = student.target_volume

    if delta_e >= -student.theta and cur > student.epsilon:
        return (
            cur
            + student.alpha * (current_neighbor_avg_volume - cur)
            + student.beta * (student.target_ref_volume - cur)
        )

    if delta_e <= -student.theta:
        return student.gamma * cur

    if delta_e >= -student.theta and cur <= student.epsilon:
        return cur + student.lambda_rate * (student.target_ref_volume - cur)

    return student.target_volume


def update_student_state(
    student: Student,
    students_map: Dict[Tuple[int, int], Student],
    p_on: float,
    p_off: float,
):
    new_status = on_off_model(student, p_on, p_off)
    current_neighbor_avg = get_neighbors_volume(student, students_map)
    hypothetical = replace(student, status=new_status)
    new_volume = update_student_volume(hypothetical, current_neighbor_avg)
    return {
        "new_status": int(new_status),
        "new_target_volume": float(new_volume),
        "new_prev_neighbor_avg_volume": float(current_neighbor_avg),
    }


def run_simulation(params: dict) -> pd.DataFrame:
    if params.get("seed") is not None:
        np.random.seed(params["seed"])

    rows = int(params["row_num"])
    cols = int(params["col_num"])
    time_steps = int(params["time_steps"])
    p_on = float(params["p_on"])
    p_off = float(params["p_off"])

    students_map: Dict[Tuple[int, int], Student] = {}
    for i in range(rows):
        for j in range(cols):
            students_map[(i, j)] = Student(
                coord=(i, j),
                status=1,
                sensitivity=1.0,
                alpha=float(params["alpha"]),
                beta=float(params["beta"]),
                gamma=float(params["gamma"]),
                lambda_rate=float(params["lambda_rate"]),
                epsilon=float(params["epsilon"]),
                theta_0=float(params["theta_0"]),
                target_ref_volume=float(params["target_ref_volume"]),
                target_volume=1.0,
                prev_neighbor_avg_volume=0.0,
            )

    history = []
    for t in range(time_steps):
        updates = {}
        for coord, student in students_map.items():
            updates[coord] = update_student_state(student, students_map, p_on, p_off)

        for coord, info in updates.items():
            s = students_map[coord]
            s.status = info["new_status"]
            s.target_volume = info["new_target_volume"]
            s.prev_neighbor_avg_volume = info["new_prev_neighbor_avg_volume"]

        for s in students_map.values():
            history.append(
                {
                    "time_step": t,
                    "coord": s.coord,
                    "status": int(s.status),
                    "target_volume": float(s.target_volume),
                    "actual_volume": float(s.actual_volume),
                }
            )

    return pd.DataFrame(history)
