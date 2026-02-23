import matplotlib.pyplot as plt
import re
import numpy as np

# Robot spawn positions (tb1~tb4)
robot_spawns = {
    'tb1': (-3.5, 1.5),
    'tb2': (4.5, 1.5),
    'tb3': (-1.5, -0.5),
    'tb4': (1.0, -8.0),
}

# Charger positions
chargers = [
    {'id': 1, 'pos': (0.0, 4.0)},
    {'id': 2, 'pos': (0.0, -6.0)},
]

# Task type color mapping
color_map = {
    'one_way': '#3399ff',      # blue
    'round_trip': '#33cc33',   # green
    'via_point': '#ffd700',    # yellow
    'move': '#888888',         # gray
    'charge': '#ff66cc',       # pink
}

# Parse the simulation log
def parse_simulation_log(log_path):
    tasks = []
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    task = None
    for line in lines:
        if line.startswith('Task'):
            if task:
                tasks.append(task)
            m = re.match(r'Task (\d+) \(([^)]+)\):', line)
            if m:
                task = {
                    'id': int(m.group(1)),
                    'type': m.group(2),
                    'segments': [],
                    'total_time': 0.0,
                    'total_energy': 0.0,
                }
        elif line.strip().startswith('drive'):
            m = re.match(r'\s*drive (\d+): \(([^,]+), ([^\)]+)\)->\(([^,]+), ([^\)]+)\) dist=([\d.]+)m t=([\d.]+)s E=([\d.]+)Wh', line)
            if m:
                seg = {
                    'type': 'drive',
                    'from': (float(m.group(2)), float(m.group(3))),
                    'to': (float(m.group(4)), float(m.group(5))),
                    'dist': float(m.group(6)),
                    'time': float(m.group(7)),
                    'energy': float(m.group(8)),
                }
                task['segments'].append(seg)
        elif line.strip().startswith('pick') or line.strip().startswith('drop'):
            # Not visualized as bars, but could be annotated
            pass
        elif line.strip().startswith('TOTAL:'):
            m = re.match(r'\s*TOTAL: t=([\d.]+)s, E=([\d.]+)Wh', line)
            if m:
                task['total_time'] = float(m.group(1))
                task['total_energy'] = float(m.group(2))
    if task:
        tasks.append(task)
    return tasks

def euclidean(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

from typing import Any, Dict, List, Optional, Tuple, Union
import heapq
import matplotlib.pyplot as plt


from typing import Any, Dict, List, Optional, Tuple, Union
import heapq
import matplotlib.pyplot as plt


def plot_robot_timelines_single(
    tasks: List[Dict[str, Any]],
    robot_sequences: Dict[str, List[Union[int, str]]],
    robot_spawns: Dict[str, Tuple[float, float]],
    chargers: List[Dict[str, Any]],
    color_map: Dict[str, str],
    algo_name: str = "RIME",
    target_makespan: Optional[Dict[str, float]] = None,
) -> None:
    robot_order: List[str] = ["tb1", "tb2", "tb3", "tb4"]
    robots: List[str] = [r for r in robot_order if r in robot_sequences]

    # params
    robot_cap_wh_map: Dict[str, float] = {"tb1": 40.0, "tb2": 60.0, "tb3": 40.0, "tb4": 60.0}
    charge_rate_wh_per_s: float = 0.1
    move_speed: float = 0.3
    K_MOVE_WH_PER_M: float = 0.275  # logging only

    tasks_by_id: Dict[int, Dict[str, Any]] = {t["id"]: t for t in tasks}
    charger_pos_by_id: Dict[int, Tuple[float, float]] = {c["id"]: c["pos"] for c in chargers}

    def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx * dx + dy * dy) ** 0.5

    def token_to_charger_id(tok: Union[int, str]) -> Optional[int]:
        if tok == "c1":
            return 1
        if tok == "c2":
            return 2
        return None

    # -------------------------------
    # Core simulation + charging schedule
    # -------------------------------
    def simulate_with_scales(scales: Dict[str, float]) -> Tuple[
        Dict[str, List[Dict[str, Any]]],  # segments_by_robot
        Dict[str, Dict[str, float]],      # stats
        Dict[str, float],                 # makespan_by_robot
        List[Dict[str, Any]],             # charge_records (global)
    ]:
        segments_by_robot: Dict[str, List[Dict[str, Any]]] = {r: [] for r in robots}
        stats: Dict[str, Dict[str, float]] = {
            r: {
                "move_time": 0.0,
                "charge_wait": 0.0,
                "charge_time": 0.0,
                "task_time": 0.0,      # already scaled
                "task_energy": 0.0,
                "move_energy": 0.0,
            }
            for r in robots
        }

        # per-robot state
        state: Dict[str, Dict[str, Any]] = {}
        for r in robots:
            cap = robot_cap_wh_map.get(r, 40.0)
            state[r] = {
                "seq": robot_sequences[r],
                "idx": 0,
                "time": 0.0,
                "pos": robot_spawns[r],
                "soc": cap,
                "cap": cap,
            }

        # charger availability per physical charger
        charger_free: Dict[int, float] = {1: 0.0, 2: 0.0}

        def push_seg(robot: str, seg: Dict[str, Any]) -> None:
            segments_by_robot[robot].append(seg)

        def advance_until_charge(robot: str) -> Optional[Dict[str, Any]]:
            st = state[robot]
            seq = st["seq"]
            s = float(scales.get(robot, 1.0))

            while st["idx"] < len(seq):
                tid = seq[st["idx"]]

                cid = token_to_charger_id(tid)
                if cid is not None or tid == -1:
                    prev_pos = st["pos"]
                    if cid is None:
                        cid = min(charger_pos_by_id.keys(), key=lambda k: euclidean(prev_pos, charger_pos_by_id[k]))
                    charger_pos = charger_pos_by_id[cid]

                    move_dist = euclidean(prev_pos, charger_pos)
                    move_time = move_dist / move_speed if move_speed > 1e-9 else 0.0
                    if move_time > 0.0:
                        push_seg(robot, {"start": st["time"], "dur": move_time, "kind": "move"})
                        st["time"] += move_time
                        stats[robot]["move_time"] += move_time
                        stats[robot]["move_energy"] += K_MOVE_WH_PER_M * move_dist

                    st["pos"] = charger_pos
                    arrival_time = st["time"]

                    remaining_wh = max(0.0, st["cap"] - st["soc"])
                    charge_time = remaining_wh / charge_rate_wh_per_s if charge_rate_wh_per_s > 1e-12 else 0.0

                    st["idx"] += 1
                    return {"robot": robot, "charger_id": int(cid), "arrival": arrival_time, "charge_time": charge_time}

                # normal task
                st["idx"] += 1
                if not isinstance(tid, int):
                    continue
                task = tasks_by_id.get(int(tid), None)
                if task is None:
                    continue

                prev_pos = st["pos"]
                task_start = task["segments"][0]["from"] if task["segments"] else prev_pos

                move_dist = euclidean(prev_pos, task_start)
                move_time = move_dist / move_speed if move_speed > 1e-9 else 0.0
                if move_time > 0.0:
                    push_seg(robot, {"start": st["time"], "dur": move_time, "kind": "move"})
                    st["time"] += move_time
                    stats[robot]["move_time"] += move_time
                    stats[robot]["move_energy"] += K_MOVE_WH_PER_M * move_dist

                # SCALE HERE (before scheduling)
                main_time_raw = float(task.get("total_time", 0.0))
                main_time = main_time_raw * s

                push_seg(
                    robot,
                    {
                        "start": st["time"],
                        "dur": main_time,
                        "kind": "task",
                        "type": task.get("type", "unknown"),
                        "task_id": int(tid),
                    },
                )
                st["time"] += main_time
                stats[robot]["task_time"] += main_time

                st["pos"] = task["segments"][-1]["to"] if task["segments"] else task_start

                e = float(task.get("total_energy", 0.0))
                st["soc"] = max(0.0, st["soc"] - e)
                stats[robot]["task_energy"] += e

            return None

        # init charge requests heap
        heap: List[Tuple[float, str, Dict[str, Any]]] = []
        for r in robots:
            req = advance_until_charge(r)
            if req is not None:
                heapq.heappush(heap, (float(req["arrival"]), r, req))

        charge_records: List[Dict[str, Any]] = []

        # schedule charges
        while heap:
            arrival, robot, req = heapq.heappop(heap)
            cid = int(req["charger_id"])

            start = arrival if charger_free[cid] <= arrival else charger_free[cid]
            wait = start - arrival
            end = start + float(req["charge_time"])

            if wait > 1e-9:
                push_seg(robot, {"start": arrival, "dur": wait, "kind": "charge_wait", "charger_id": cid})
                stats[robot]["charge_wait"] += wait

            if float(req["charge_time"]) > 1e-9:
                push_seg(robot, {"start": start, "dur": float(req["charge_time"]), "kind": "charge", "charger_id": cid})
                stats[robot]["charge_time"] += float(req["charge_time"])

            charge_records.append(
                {"charger_id": cid, "robot": robot, "arrival": arrival, "start": start, "end": end, "wait": wait, "charge": float(req["charge_time"])}
            )

            charger_free[cid] = end

            st = state[robot]
            st["time"] = end
            st["soc"] = st["cap"]

            next_req = advance_until_charge(robot)
            if next_req is not None:
                heapq.heappush(heap, (float(next_req["arrival"]), robot, next_req))

        # rebuild contiguous starts per robot (append order is already chronological for that robot)
        makespan_by_robot: Dict[str, float] = {}
        for r in robots:
            tcur = 0.0
            new_segs: List[Dict[str, Any]] = []
            for seg in segments_by_robot[r]:
                new_seg = dict(seg)
                new_seg["start"] = tcur
                new_segs.append(new_seg)
                tcur += float(new_seg["dur"])
            segments_by_robot[r] = new_segs
            makespan_by_robot[r] = tcur

        return segments_by_robot, stats, makespan_by_robot, charge_records

    # -------------------------------
    # Fit scales to target makespan (iterate)
    # -------------------------------
    scales: Dict[str, float] = {r: 1.0 for r in robots}

    if target_makespan is not None:
        for _ in range(15):  # iterations
            segs, st, ms, _ = simulate_with_scales(scales)

            # update scales using task portion only (keep move/charge physics fixed)
            max_err = 0.0
            for r in robots:
                tgt = float(target_makespan[r])
                fixed = st[r]["move_time"] + st[r]["charge_wait"] + st[r]["charge_time"]
                task_time = st[r]["task_time"]
                cur = float(ms[r])

                # if task_time is ~0, skip
                if task_time <= 1e-9:
                    continue

                # desired task_time to meet target
                desired_task = max(0.0, tgt - fixed)
                # multiplicative update
                ratio = desired_task / task_time if task_time > 1e-9 else 1.0
                # damped update to avoid oscillation
                scales[r] = max(0.0, scales[r] * (0.7 * ratio + 0.3))

                max_err = max(max_err, abs(cur - tgt))

            if max_err < 1e-2:
                break

    # final run with fitted scales (or 1.0 if no target)
    segments_by_robot, stats, makespan_by_robot, charge_records = simulate_with_scales(scales)

    # -------------------------------
    # DEBUG LOG: verify wait end == some charge end on same charger
    # -------------------------------
    print(f"[Timeline] algo={algo_name}")
    for r in robots:
        print(
            f"  {r}: scale={scales[r]:.6f} makespan={makespan_by_robot[r]:.3f}s"
            + (f" target={target_makespan[r]:.3f}s" if target_makespan and r in target_makespan else "")
            + f" energy_used(task+move)={(stats[r]['task_energy'] + stats[r]['move_energy']):.3f}Wh"
        )


    # Optional: print charge schedule table (charger contention check)
    # for rec in charge_records:
    #     print(f"  c{rec['charger_id']} {rec['robot']} arrival={rec['arrival']:.2f} start={rec['start']:.2f} end={rec['end']:.2f} wait={rec['wait']:.2f}")

    # -------------------------------
    # RENDER
    # -------------------------------
    fig, ax = plt.subplots(figsize=(18, 5))

    for ridx, r in enumerate(robots):
        for seg in segments_by_robot[r]:
            kind = str(seg["kind"])
            left = float(seg["start"])
            width = float(seg["dur"])
            if width <= 0.0:
                continue

            if kind == "move":
                ax.barh(ridx, width, left=left, color=color_map["move"], edgecolor="k", height=0.7, linewidth=1)
            
            elif kind == "charge_wait":
                ax.barh(
                    ridx, width, left=left,
                    color="black", edgecolor="k",
                    height=0.7, linewidth=1
                )

            elif kind == "charge":
                ax.barh(
                    ridx, width, left=left,
                    color="white", edgecolor="k",
                    height=0.7, linewidth=1,
                    hatch="//"
                )
                cid = seg.get("charger_id", None)
                if cid is not None and width > 2.0:
                    ax.text(
                        left + width / 2.0, ridx,
                        f"c{cid}",
                        va="center", ha="center",
                        fontsize=12, color="black"
                    )      
            elif kind == "task":
                ttype = str(seg.get("type", "unknown"))
                ax.barh(ridx, width, left=left, color=color_map.get(ttype, "#cccccc"), edgecolor="k", height=0.7, linewidth=1)
                tid = seg.get("task_id", None)
                if tid is not None and width > 2.0:
                    ax.text(left + width / 2.0, ridx, f"{tid}", va="center", ha="center", fontsize=9, color="black")

    ax.set_yticks(range(len(robots)))
    ax.set_yticklabels(robots)
    ax.invert_yaxis()  # tb1 on top, tb4 bottom

    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_title(f"Task Timeline ({algo_name})", fontsize=16)

    proxy = {
        "Travel": plt.Rectangle((0, 0), 1, 1, facecolor=color_map["move"], edgecolor="k", linewidth=1),
        "One-way": plt.Rectangle((0, 0), 1, 1, facecolor=color_map["one_way"], edgecolor="k", linewidth=1),
        "Round-trip": plt.Rectangle((0, 0), 1, 1, facecolor=color_map["round_trip"], edgecolor="k", linewidth=1),
        "Via-point": plt.Rectangle((0, 0), 1, 1, facecolor=color_map["via_point"], edgecolor="k", linewidth=1),
        "Charge": plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="k", hatch="//", linewidth=1),
        "Charge wait": plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="k", linewidth=1),
    }

    ax.legend(
        [proxy[k] for k in ["Travel", "One-way", "Round-trip", "Via-point", "Charge", "Charge wait"]],
        ["Travel", "One-way", "Round-trip", "Via-point", "Charge", "Charge wait"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=6,
        frameon=False,
        fontsize=13,
        handlelength=2.2,
        columnspacing=1.6,
        handletextpad=0.6,
    )

    plt.tight_layout(rect=[0, 0.14, 1, 1])
    plt.show()


if __name__ == "__main__":
    log_path = "task_time_energy_simulation_log.txt"
    tasks = parse_simulation_log(log_path)
    robot_sequences = {
        'tb1': [6,38,1,'c1',26,18,10,27,49,'c1',33,'c2',15,47,37,17],
        'tb2': [21,34,23,'c2',48,43,19,42,11,13,3,35,9,39],
        'tb3': [14,4,25,24,'c1',20,46,8,'c2',31,30,32,44],
        'tb4': [36,16,'c2',28,40,22,'c2',7,12,45,41,2,50,5,29],
    }
    target_makespan = {
        'tb1': 1492.98,
        'tb2': 1197.97,
        'tb3': 1497.54,
        'tb4': 1490.13,
    }

    plot_robot_timelines_single(
    tasks, robot_sequences, robot_spawns, chargers, color_map,
    algo_name="RIME",
    target_makespan=target_makespan,
    )
