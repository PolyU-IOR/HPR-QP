#!/usr/bin/env python3

import argparse
import csv
import math
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CSV_NAME = "HPRQP_result.csv"
LOG_NAME = "HPRQP_log.txt"
SUMMARY_NAMES = {"SGM10", "solved"}
SUCCESS_STATUSES = {"OK", "MAX_ITER", "TIME_LIMIT"}
FIELDNAMES = [
    "name",
    "iter",
    "alg_time",
    "res",
    "primal_obj",
    "gap",
    "status",
    "iter_4",
    "time_4",
    "iter_6",
    "time_6",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run solve_qp on all .mps files in a folder.")
    parser.add_argument("data_path", help="Folder containing .mps files")
    parser.add_argument("result_path", help="Folder for CSV and log outputs")
    parser.add_argument("--build-dir", default="build", help="Build directory containing solve_qp")
    parser.add_argument("--solver", default="solve_qp", help="Solver executable name")
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--stoptol", type=float)
    parser.add_argument("--check-iter", type=int)
    parser.add_argument("--print-frequency", type=int)
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--verbose", type=int)
    parser.add_argument("--use-gpu", type=int)
    parser.add_argument("--device-number", type=int)
    parser.add_argument("--spmv-mode-q", choices=["auto", "cusparse", "customized"])
    parser.add_argument("--spmv-mode-a", choices=["auto", "cusparse", "customized"])
    parser.add_argument("--use-scaling", type=int)
    return parser.parse_args()


def solver_command(args: argparse.Namespace, mps_path: Path) -> list[str]:
    command = [str(Path(args.build_dir) / args.solver), str(mps_path)]
    optional_args = {
        "--max-iter": args.max_iter,
        "--stoptol": args.stoptol,
        "--check-iter": args.check_iter,
        "--print-frequency": args.print_frequency,
        "--time-limit": args.time_limit,
        "--verbose": args.verbose,
        "--use-gpu": args.use_gpu,
        "--device-number": args.device_number,
        "--use-scaling": args.use_scaling,
        "--spmv-mode-q": args.spmv_mode_q,
        "--spmv-mode-a": args.spmv_mode_a,
    }
    for flag, value in optional_args.items():
        if value is not None:
            command.extend([flag, str(value)])
    return command


def parse_solver_output(output: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def load_existing_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("name", "") not in SUMMARY_NAMES]
    return rows


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="ascii") as handle:
        handle.write(message)


def parse_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(row: dict[str, str], key: str) -> int | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def shifted_geomean(values: list[float], shift: float = 10.0) -> str:
    if not values:
        return ""
    return f"{math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift:.12g}"


def write_csv(csv_path: Path, rows: list[dict[str, str]], time_limit: float) -> None:
    data_rows = list(rows)
    alg_times = [value for row in data_rows if (value := parse_float(row, "alg_time")) is not None]
    iter_values = [float(value) for row in data_rows if (value := parse_int(row, "iter")) is not None]
    time4_values = [value for row in data_rows if (value := parse_float(row, "time_4")) is not None]
    iter4_values = [float(value) for row in data_rows if (value := parse_int(row, "iter_4")) is not None]
    time6_values = [value for row in data_rows if (value := parse_float(row, "time_6")) is not None]
    iter6_values = [float(value) for row in data_rows if (value := parse_int(row, "iter_6")) is not None]

    sgm_row = {
        "name": "SGM10",
        "iter": shifted_geomean(iter_values),
        "alg_time": shifted_geomean(alg_times),
        "res": "",
        "primal_obj": "",
        "gap": "",
        "status": "",
        "iter_4": shifted_geomean(iter4_values),
        "time_4": shifted_geomean(time4_values),
        "iter_6": shifted_geomean(iter6_values),
        "time_6": shifted_geomean(time6_values),
    }
    solved_row = {
        "name": "solved",
        "iter": "",
        "alg_time": str(sum(1 for value in alg_times if value < time_limit)),
        "res": "",
        "primal_obj": "",
        "gap": "",
        "status": "",
        "iter_4": "",
        "time_4": str(sum(1 for value in time4_values if value < time_limit)),
        "iter_6": "",
        "time_6": str(sum(1 for value in time6_values if value < time_limit)),
    }

    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(data_rows)
        writer.writerow(sgm_row)
        writer.writerow(solved_row)


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    result_path = Path(args.result_path)
    solver_path = Path(args.build_dir) / args.solver

    if not data_path.is_dir():
        print(f"dataset folder not found: {data_path}", file=sys.stderr)
        return 1
    if not solver_path.exists():
        print(f"solver executable not found: {solver_path}", file=sys.stderr)
        return 1

    result_path.mkdir(parents=True, exist_ok=True)
    csv_path = result_path / CSV_NAME
    log_path = result_path / LOG_NAME

    rows = load_existing_rows(csv_path)
    row_index_by_name = {row["name"]: index for index, row in enumerate(rows) if row.get("name")}
    completed = {
        row["name"]
        for row in rows
        if row.get("name") and row.get("status", "") in SUCCESS_STATUSES
    }
    files = sorted(path for path in data_path.iterdir() if path.is_file() and path.suffix.lower() == ".mps")

    for index, mps_path in enumerate(files, start=1):
        if mps_path.name in completed:
            continue

        command = solver_command(args, mps_path)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        banner = f"[{stamp}] solving problem {index}: {mps_path.name}\n"
        command_line = f"command: {' '.join(shlex.quote(part) for part in command)}\n"
        print(banner.strip())

        completed_process = subprocess.run(command, capture_output=True, text=True)
        combined_output = completed_process.stdout
        if completed_process.stderr:
            combined_output = combined_output + ("\n" if combined_output and not combined_output.endswith("\n") else "") + completed_process.stderr

        append_log(log_path, banner)
        append_log(log_path, command_line)
        append_log(log_path, combined_output)
        if combined_output and not combined_output.endswith("\n"):
            append_log(log_path, "\n")

        parsed = parse_solver_output(combined_output)
        row = {
            "name": mps_path.name,
            "iter": parsed.get("iter", ""),
            "alg_time": parsed.get("time", ""),
            "res": parsed.get("residual", ""),
            "primal_obj": parsed.get("objective", ""),
            "gap": parsed.get("gap", ""),
            "status": parsed.get("status", f"EXIT_{completed_process.returncode}"),
            "iter_4": parsed.get("iter_4", ""),
            "time_4": parsed.get("time_4", ""),
            "iter_6": parsed.get("iter_6", ""),
            "time_6": parsed.get("time_6", ""),
        }
        existing_index = row_index_by_name.get(mps_path.name)
        if existing_index is None:
            row_index_by_name[mps_path.name] = len(rows)
            rows.append(row)
        else:
            rows[existing_index] = row

        if row["status"] in SUCCESS_STATUSES:
            completed.add(mps_path.name)
        else:
            completed.discard(mps_path.name)
        write_csv(csv_path, rows, args.time_limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())