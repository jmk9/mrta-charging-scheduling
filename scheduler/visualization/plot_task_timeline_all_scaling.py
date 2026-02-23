import re
import heapq
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt


# -------------------------
# Fixed configs
# -------------------------
robot_spawns: Dict[str, Tuple[float, float]] = {
    "tb1": (-3.5, 1.5),
    "tb2": (4.5, 1.5),
    "tb3": (-1.5, -0.5),
    "tb4": (1.0, -8.0),
}

display_name_map: Dict[str, str] = {
    "tb1": "r1",
    "tb2": "r2",
    "tb3": "r3",
    "tb4": "r4",
}

chargers: List[Dict[str, Any]] = [
    {"id": 1, "pos": (0.0, -6.0)},
    {"id": 2, "pos": (0.0, 4.0)},
]

color_map: Dict[str, str] = {
    "one_way": "#3399ff",
    "round_trip": "#33cc33",
    "via_point": "#ffb347",
    "move": "#888888",
    "charge": "#ff66cc",
}

# Set this to a float (e.g. 1.128267) to apply one global scale everywhere.
# Leave as None to auto-compute the ratio from SMA tb3.
GLOBAL_SCALE_OVERRIDE: Optional[float] = 1.38


# -------------------------
# Parse the simulation log
# -------------------------
def parse_simulation_log(log_path: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    task: Optional[Dict[str, Any]] = None
    for line in lines:
        if line.startswith("Task"):
            if task:
                tasks.append(task)
            m = re.match(r"Task (\d+) \(([^)]+)\):", line)
            if m:
                task = {
                    "id": int(m.group(1)),
                    "type": m.group(2),
                    "segments": [],
                    "total_time": 0.0,
                    "total_energy": 0.0,
                }

        elif line.strip().startswith("drive"):
            m = re.match(
                r"\s*drive (\d+): \(([^,]+), ([^\)]+)\)->\(([^,]+), ([^\)]+)\) dist=([\d.]+)m t=([\d.]+)s E=([\d.]+)Wh",
                line,
            )
            if m and task is not None:
                seg = {
                    "type": "drive",
                    "from": (float(m.group(2)), float(m.group(3))),
                    "to": (float(m.group(4)), float(m.group(5))),
                    "dist": float(m.group(6)),
                    "time": float(m.group(7)),
                    "energy": float(m.group(8)),
                }
                task["segments"].append(seg)

        elif line.strip().startswith("TOTAL:"):
            m = re.match(r"\s*TOTAL: t=([\d.]+)s, E=([\d.]+)Wh", line)
            if m and task is not None:
                task["total_time"] = float(m.group(1))
                task["total_energy"] = float(m.group(2))

    if task:
        tasks.append(task)
    return tasks


# -------------------------
# Timeline core
# -------------------------
def plot_one_timeline_on_ax(
    ax: Optional[plt.Axes],
    tasks: List[Dict[str, Any]],
    robot_sequences: Dict[str, List[Union[int, str]]],
    robot_spawns_in: Dict[str, Tuple[float, float]],
    chargers_in: List[Dict[str, Any]],
    color_map_in: Dict[str, str],
    algo_name: str,
    target_makespan: Optional[Dict[str, float]] = None,
    charger_label_fontsize: int = 12,
    override_scales: Optional[Dict[str, float]] = None,
    render: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, float], Dict[str, Dict[str, float]]]:
    robot_order: List[str] = ["tb1", "tb2", "tb3", "tb4"]
    robots: List[str] = [r for r in robot_order if r in robot_sequences]
    display_labels: List[str] = [display_name_map.get(r, r) for r in robots]

    # params (same as your current assumptions)
    robot_cap_wh_map: Dict[str, float] = {"tb1": 40.0, "tb2": 60.0, "tb3": 40.0, "tb4": 60.0}
    charge_rate_wh_per_s: float = 0.1
    move_speed: float = 0.3
    K_MOVE_WH_PER_M: float = 0.275  # logging only

    tasks_by_id: Dict[int, Dict[str, Any]] = {t["id"]: t for t in tasks}
    charger_pos_by_id: Dict[int, Tuple[float, float]] = {c["id"]: c["pos"] for c in chargers_in}

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

    def simulate_with_scales(scales: Dict[str, float]) -> Tuple[
        Dict[str, List[Dict[str, Any]]],  # segments_by_robot
        Dict[str, Dict[str, float]],      # stats
        Dict[str, float],                 # makespan_by_robot
    ]:
        segments_by_robot: Dict[str, List[Dict[str, Any]]] = {r: [] for r in robots}
        stats: Dict[str, Dict[str, float]] = {
            r: {"move_time": 0.0, "charge_wait": 0.0, "charge_time": 0.0, "task_time": 0.0, "task_energy": 0.0, "move_energy": 0.0}
            for r in robots
        }

        state: Dict[str, Dict[str, Any]] = {}
        for r in robots:
            cap = robot_cap_wh_map.get(r, 40.0)
            state[r] = {"seq": robot_sequences[r], "idx": 0, "time": 0.0, "pos": robot_spawns_in[r], "soc": cap, "cap": cap}

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
                    charger_pos = charger_pos_by_id[int(cid)]

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

                main_time_raw = float(task.get("total_time", 0.0))
                main_time = main_time_raw * s

                push_seg(robot, {"start": st["time"], "dur": main_time, "kind": "task", "type": task.get("type", "unknown"), "task_id": int(tid)})
                st["time"] += main_time
                stats[robot]["task_time"] += main_time

                st["pos"] = task["segments"][-1]["to"] if task["segments"] else task_start

                e = float(task.get("total_energy", 0.0))
                st["soc"] = max(0.0, st["soc"] - e)
                stats[robot]["task_energy"] += e

            return None

        heap: List[Tuple[float, str, Dict[str, Any]]] = []
        for r in robots:
            req = advance_until_charge(r)
            if req is not None:
                heapq.heappush(heap, (float(req["arrival"]), r, req))

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

            charger_free[cid] = end

            st = state[robot]
            st["time"] = end
            st["soc"] = st["cap"]

            nxt = advance_until_charge(robot)
            if nxt is not None:
                heapq.heappush(heap, (float(nxt["arrival"]), robot, nxt))

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

        return segments_by_robot, stats, makespan_by_robot

    # --- scale fitting ---
    if override_scales is not None:
        scales: Dict[str, float] = {r: float(override_scales.get(r, 1.0)) for r in robots}
    else:
        scales = {r: 1.0 for r in robots}
        if target_makespan is not None:
            for _ in range(15):
                segs, st, ms = simulate_with_scales(scales)
                max_err = 0.0
                for r in robots:
                    tgt = float(target_makespan[r])
                    fixed = st[r]["move_time"] + st[r]["charge_wait"] + st[r]["charge_time"]
                    task_time = st[r]["task_time"]
                    cur = float(ms[r])
                    if task_time <= 1e-9:
                        continue
                    desired_task = max(0.0, tgt - fixed)
                    ratio = desired_task / task_time if task_time > 1e-9 else 1.0
                    scales[r] = max(0.0, scales[r] * (0.7 * ratio + 0.3))
                    max_err = max(max_err, abs(cur - tgt))
                if max_err < 1e-2:
                    break

    segments_by_robot, stats, makespan_by_robot = simulate_with_scales(scales)

    # --- print per-algo log ---
    if verbose:
        print(f"[Timeline] algo={algo_name}")
        for r, label in zip(robots, display_labels):
            tgt = float(target_makespan[r]) if target_makespan and r in target_makespan else None
            energy_used = stats[r]["task_energy"] + stats[r]["move_energy"]
            msg = f"  {label}: scale={scales[r]:.6f} makespan={makespan_by_robot[r]:.3f}s"
            if tgt is not None:
                msg += f" target={tgt:.3f}s"
            msg += f" energy_used(task+move)={energy_used:.3f}Wh"
            # 추가: travel/charge/wait 시간도 함께 출력
            msg += f" travel={stats[r]['move_time']:.2f}s, charge={stats[r]['charge_time']:.2f}s, wait={stats[r]['charge_wait']:.2f}s"
            print(msg)

    aggregate_totals = {
        "travel_time": sum(stats[r]["move_time"] for r in robots),
        "charge_time": sum(stats[r]["charge_time"] for r in robots),
        "charge_wait": sum(stats[r]["charge_wait"] for r in robots),
    }

    if render and ax is not None:
        # Find the robot with the maximum makespan for this algorithm
        max_makespan = max(makespan_by_robot.values()) if makespan_by_robot else None
        max_robots = [r for r in robots if abs(makespan_by_robot[r] - max_makespan) < 1e-6] if max_makespan is not None else []

        for ridx, r in enumerate(robots):
            for seg in segments_by_robot[r]:
                kind = str(seg["kind"])
                left = float(seg["start"])
                width = float(seg["dur"])
                if width <= 0.0:
                    continue

                if kind == "move":
                    ax.barh(ridx, width, left=left, color=color_map_in["move"], edgecolor="k", height=0.6, linewidth=1)

                elif kind == "charge_wait":
                    ax.barh(ridx, width, left=left, color="black", edgecolor="k", height=0.6, linewidth=1)

                elif kind == "charge":
                    ax.barh(ridx, width, left=left, color="white", edgecolor="k", height=0.6, linewidth=1, hatch="//")
                    cid = seg.get("charger_id", None)
                    if cid is not None and width > 2.0:
                        ax.text(
                            left + width / 2.0,
                            ridx,
                            f"c{cid}",
                            va="center",
                            ha="center",
                            fontsize=charger_label_fontsize,
                            color="black",
                        )

                elif kind == "task":
                    ttype = str(seg.get("type", "unknown"))
                    ax.barh(ridx, width, left=left, color=color_map_in.get(ttype, "#cccccc"), edgecolor="k", height=0.6, linewidth=1)
                    tid = seg.get("task_id", None)
                    if tid is not None and width > 2.0:
                        ax.text(left + width / 2.0, ridx, f"{tid}", va="center", ha="center", fontsize=9, color="black")

            # Add makespan label at the right end of the bar
            makespan = makespan_by_robot[r]
            # Bold only for the robot(s) with the maximum makespan, thin for others
            fontweight = "bold" if r in max_robots else "normal"
            ax.text(
                makespan + 5,  # small offset to the right
                ridx,
                f"{makespan:.1f}s",
                va="center",
                ha="left",
                fontsize=13,
                color="black",
                fontweight=fontweight,
            )

        ax.set_yticks(range(len(robots)))
        ax.set_yticklabels(display_labels, fontsize=14)
        ax.tick_params(axis="both", labelsize=13)
        ax.invert_yaxis()
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_title(f"Task Timeline ({algo_name})", fontsize=18)

    algo_makespan = max(makespan_by_robot.values()) if makespan_by_robot else 0.0
    return aggregate_totals, scales, algo_makespan, makespan_by_robot, stats


def make_6_timelines_figure(
    tasks: List[Dict[str, Any]],
    cases: List[Dict[str, Any]],
    save_path: str = "timeline_6algos.png",
) -> None:
    n = len(cases)

    # 1열로 쭉 나열
    fig, axes = plt.subplots(n, 1, figsize=(20, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]  # type: ignore

    algo_totals: List[Dict[str, Any]] = []
    algo_max_spans: List[float] = []

    for ax, case in zip(axes, cases):
        case_override = case.get("override_scales")  # allow per-case overrides if provided
        totals, _, algo_ms, _, _ = plot_one_timeline_on_ax(
            ax=ax,
            tasks=tasks,
            robot_sequences=case["robot_sequences"],
            robot_spawns_in=robot_spawns,
            chargers_in=chargers,
            color_map_in=color_map,
            algo_name=case["algo_name"],
            target_makespan=case["target_makespan"],
            charger_label_fontsize=12,
            override_scales=case_override,
        )
        algo_totals.append({
            "algo_name": case["algo_name"],
            "travel_time": totals["travel_time"],
            "charge_time": totals["charge_time"],
            "charge_wait": totals["charge_wait"],
        })
        algo_max_spans.append(algo_ms)

    # ---- unify x-axis length across all subplots ----
    # Force x-axis to 0~1700 for all subplots
    for ax in axes:
        ax.set_xlim(0.0, 1685.0)

    # global legend: 맨 아래 1번만
    proxy = {
        "Travel": plt.Rectangle((0, 0), 1, 1, facecolor=color_map["move"], edgecolor="k", linewidth=1),
        "One-way": plt.Rectangle((0, 0), 1, 1, facecolor=color_map["one_way"], edgecolor="k", linewidth=1),
        "Round-trip": plt.Rectangle((0, 0), 1, 1, facecolor=color_map["round_trip"], edgecolor="k", linewidth=1),
        "Via-point": plt.Rectangle((0, 0), 1, 1, facecolor=color_map["via_point"], edgecolor="k", linewidth=1),
        "Charge wait": plt.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="k", linewidth=1),
        "Charge": plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="k", hatch="//", linewidth=1)
    }

    fig.legend(
        [proxy[k] for k in ["Travel", "One-way", "Round-trip", "Via-point", "Charge wait", "Charge"]],
        ["Travel", "One-way", "Round-trip", "Via-point", "Charge wait", "Charge"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6,
        frameon=False,
        fontsize=22,
        handlelength=2.2,
        columnspacing=1.6,
        handletextpad=0.6,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"[Saved] {save_path}")

    print("[Total Time Summary Across Algorithms]")
    for entry in algo_totals:
        print(
            f"  {entry['algo_name']}: travel={entry['travel_time']:.2f}s, "
            f"charge={entry['charge_time']:.2f}s, wait={entry['charge_wait']:.2f}s"
        )


def compute_sma_tb3_ratio(tasks: List[Dict[str, Any]], cases: List[Dict[str, Any]]) -> Optional[float]:
    reference_case = next((case for case in cases if case["algo_name"].lower() == "sma"), None)
    if reference_case is None:
        return None

    target_map = reference_case.get("target_makespan", {})
    if "tb3" not in target_map:
        return None

    base_override = {r: 1.0 for r in reference_case["robot_sequences"].keys()}
    _, _, _, robot_makespans, stats_map = plot_one_timeline_on_ax(
        ax=None,
        tasks=tasks,
        robot_sequences=reference_case["robot_sequences"],
        robot_spawns_in=robot_spawns,
        chargers_in=chargers,
        color_map_in=color_map,
        algo_name="SMA_BASELINE",
        target_makespan=None,
        charger_label_fontsize=12,
        override_scales=base_override,
        render=False,
        verbose=False,
    )

    stats_tb3 = stats_map.get("tb3")
    if stats_tb3 is None:
        return None

    task_time = stats_tb3.get("task_time", 0.0)
    fixed_time = stats_tb3.get("move_time", 0.0) + stats_tb3.get("charge_wait", 0.0) + stats_tb3.get("charge_time", 0.0)
    target_ms = float(target_map["tb3"])

    if task_time <= 1e-9:
        return None

    desired_task_time = max(0.0, target_ms - fixed_time)
    if desired_task_time <= 0.0:
        return None

    ratio = desired_task_time / task_time
    return ratio


if __name__ == "__main__":
    log_path = "task_time_energy_simulation_log.txt"
    tasks = parse_simulation_log(log_path)

    # -------------------------
    # CASES: order = PSO, WCA, QSA, SMA, FLA, RIME
    # -------------------------
    cases: List[Dict[str, Any]] = []

    # PSO
    cases.append({
        "algo_name": "PSO",
        "robot_sequences": {
            "tb1": [46, 30, "c1", 6, 41, 44, 7, 36, "c2", 22, 47, 49],
            "tb2": [48, 27, 35, 18, 32, "c1", 26, 31, "c1", 19, 16, 1, "c1", 14, 5, 50],
            "tb3": [37, 42, 9, 28, "c2", 25, 17, 45, 2, "c1", 33, 13, 29, 12, 43],
            "tb4": [20, 15, 34, 24, 11, 40, 38, "c2", 10, 39, 3, 21, 4, 8, 23],
        },
        "target_makespan": {
            "tb1": 1740.85,
            "tb2": 1739.56,
            "tb3": 1465.91,
            "tb4": 1550.86,
        }
    })

    # WCA (-1 allowed)
    cases.append({
        "algo_name": "WCA",
        "robot_sequences": {
            "tb1": [33, 34, 7, "c2", 48, 44, 1, 25, "c1", 31, 5, 30, 11, 14, 15],
            "tb2": [16, 26, "c2", 38, 8, 24, 39, "c2", 41, 20, 40, 23, 12, 2, 28, 32],
            "tb3": [46, 10, "c1", 47, 19, 27, 43, 21, 36],
            "tb4": [35, 13, 6, 22, 45, 18, "c1", 37, 3, 50, 4, 49, 9, 42, 17, 29],
        },
        "target_makespan": {
            "tb1": 1427.27,
            "tb2": 1434.73,
            "tb3": 721.00,
            "tb4": 1273.18,
        }
    })

    # QSA
    cases.append({
        "algo_name": "QSA",
        "robot_sequences": {
            "tb1": [14, 36, 7, "c2", 30, 8, "c1", 18, 15, 41, 19, "c1", 37, 1, "c1", 42, 31, 46, 34],
            "tb2": [10, 16, 20, 38, "c1", 48, 29, 22, "c2", 26, 13, 28, 9, 3, 32],
            "tb3": [4, 6, 45, 12, "c2", 40, 43, "c1", 25, 50, 44, 11],
            "tb4": [49, 35, 17, 39, 21, "c2", 2, 27, 24, 5, 33, 23, 47],
        },
        "target_makespan": {
            "tb1": 1708.02,
            "tb2": 1698.67,
            "tb3": 1207.35,
            "tb4": 1461.49,
        }
    })

    # SMA (-1 allowed)
    cases.append({
        "algo_name": "SMA",
        "robot_sequences": {
            "tb1": [6, 38, 1, "c1", 26, 18, 10, 27, 49, "c1", 33, "c1", 15, 47, 37, 17],
            "tb2": [21, 34, 23, "c2", 48, 43, 19, 42, 11, 13, 3, 35, 9, 39],
            "tb3": [14, 4, 25, 24, "c1", 20, 46, 8, "c2", 31, 30, 32, 44],
            "tb4": [36, 16, "c2", 28, 40, 22, "c2", 7, 12, 45, 41, 2, 50, 5, 29],
        },
        "target_makespan": {
            "tb1": 1423.61,
            "tb2": 1141.44,
            "tb3": 1429.22,
            "tb4": 1420.31,
        }
    })

    # FLA (-1 allowed)
    cases.append({
        "algo_name": "FLA",
        "robot_sequences": {
            "tb1": [24, "c1", 3, 43, 13, "c1", 14, 26, 8, 48, "c2", 10, 16, 15, 6, 23, 18, 9],
            "tb2": [50, 25, "c1", 17, 22, 38, "c2", 5, 19, 27, 20, 34, 49, 7, 29, 12],
            "tb3": [11, 36, "c2", 32, 39, 35, "c1", 33, 46, 44, 42],
            "tb4": [30, 2, 28, 40, "c2", 1, 37, 41, 45, 21, 31, 4, 47],
        },
        "target_makespan": {
            "tb1": 1502.60,
            "tb2": 1432.61,
            "tb3": 1358.05,
            "tb4": 1115.80,
        }
    })

    # RIME (your current one in message)
    cases.append({
        "algo_name": "RIME",
        "robot_sequences": {
            "tb1": [35, 17, "c2", 44, 36, 23, 14, "c2", 22, 38, "c1", 50, 43, 21, 49, 20],
            "tb2": [10, 32, 5, 4, 39, "c1", 6, 46, 11, 13, 33, 45, 47],
            "tb3": [19, 3, "c1", 24, "c1", 27, 25, 48, 9, "c1", 28, 34, 31, 7, 41],
            "tb4": [8, 15, 42, 37, "c2", 30, 2, 40, 29, 26, 12, 1, 18, 16],
        },
        "target_makespan": {
            "tb1": 1488.10,
            "tb2": 1381.33,
            "tb3": 1434.29,
            "tb4": 1098.22,
        }
    })

    global_scale_ratio = GLOBAL_SCALE_OVERRIDE if GLOBAL_SCALE_OVERRIDE is not None else compute_sma_tb3_ratio(tasks, cases)
    if global_scale_ratio is not None:
        print(f"[Scaling] Applying SMA tb3 ratio={global_scale_ratio:.6f} to all algorithms")
        for case in cases:
            robots = case["robot_sequences"].keys()
            case["override_scales"] = {r: global_scale_ratio for r in robots}
    else:
        print("[Scaling] Could not compute SMA tb3 ratio; falling back to per-case targets")

    make_6_timelines_figure(tasks, cases, save_path="timeline.png")
