"""查看 Quafu 任務結果（只查詢、不耗額度、零依賴）。

用法：
    python qres.py <taskid> [--golden qbn5_book] [--top 8]

三條檢視路徑的第一條（也是最可自動化的一條）：
    1. 本檔：直接打 qbackend/scq_task_recall/（標準庫 urllib，不需 pyquafu）
    2. pyquafu：Task(user=u).retrieve(taskid) -> ExecResult(.task_status/.counts/.probabilities)
    3. 官網 quafu.baqis.ac.cn 登入後的任務列表（人工看）

狀態碼（pyquafu results.py 的 status_map，原文照抄）：
    0 In Queue／1 Running／2 Completed／3 Canceled／4 Failed／5 Pending
★ 只有 2 是成功；3/4/5 都應該停止輪詢，不要傻等。
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import sys
import urllib.request

URL = "https://quafu.baqis.ac.cn/"
STATUS = {0: "In Queue", 1: "Running", 2: "Completed", 3: "Canceled", 4: "Failed", 5: "Pending"}
TOKENS = [pathlib.Path.home() / ".dsh" / "quafu.token",
          pathlib.Path(os.environ.get("QUAFU_TOKEN_FILE", "")) if os.environ.get("QUAFU_TOKEN_FILE") else pathlib.Path("/nonexistent")]
GOLDEN = pathlib.Path(__file__).resolve().parents[2] / "tests/golden/cudaq_golden.json"


def token() -> str:
    for p in TOKENS:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    raise SystemExit("找不到 quafu token（~/.dsh/quafu.token）")


def recall(tid: str) -> dict:
    req = urllib.request.Request(
        URL + "qbackend/scq_task_recall/", data=("task_id=" + tid).encode(),
        headers={"api_token": token(),
                 "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    tid = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 8
    d = recall(tid)
    st = d.get("status")
    print("taskid : %s" % d.get("task_id"))
    print("status : %s  (%s)" % (st, STATUS.get(st, "未知")))
    print("name   : %r" % d.get("task_name", ""))
    print("measure: %s" % d.get("measure"))
    print("qasm   : %d 行（轉譯後）" % len((d.get("openqasm") or "").splitlines()))

    raw = d.get("res")
    if not raw or raw in ("{}", "None"):
        print("")
        print("尚無結果（任務還沒跑完）。稍後再執行同一道指令即可。")
        return 2

    counts = ast.literal_eval(raw) if raw.startswith("{") else json.loads(raw)
    shots = sum(counts.values())
    print("")
    print("shots  : %d   （相異結果 %d 種）" % (shots, len(counts)))
    print("%-12s %8s %10s" % ("bitstring", "counts", "freq"))
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:top]:
        print("%-12s %8d %10.5f" % (k, v, v / shots))

    if "--golden" in sys.argv:
        case = sys.argv[sys.argv.index("--golden") + 1]
        g = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"][case]["probabilities"]
        n = len(next(iter(counts)))
        p = [0.0] * (1 << n)
        for k, v in counts.items():
            p[int(k[::-1], 2)] += v / shots      # quafu 是 big-endian（hardware/README §7.4）
        print("")
        print("對黃金向量 %s：max|dP| = %.3e" % (case, max(abs(p[i] - g[i]) for i in range(1 << n))))
        print("（取樣模擬器的合理量級 ~1/sqrt(shots)；真機應遠大於此，那是器件噪聲）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
