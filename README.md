# AMR resctrl/OMPT 実験コード

このリポジトリは、OMPT で OpenMP 並列領域を観測し、Linux resctrl を使って MBA/CAT/monitor 系の制御と計測を行う実験コードです。

## 設計方針

- 必須機能が使えない場合は、代替動作へ自動で切り替えずに失敗させる。
- `AMR_RESCTRL_RESOURCE=mba` では、resctrl MBA 制御と PMU の LLC カウンタを必須とする。
- `AMR_RESCTRL_RESOURCE=cat` または `monitor` では、対応する resctrl グループ制御を必須とする。
- 全体の仕様や実験上の前提はこの `README.md` に書き、エージェント向けの作業ルールは `AGENT.md` に分ける。

## MBA 制御

- MBA は 7 段階の resctrl グループで制御する。
- 7 段階は `20%, 30%, 40%, 50%, 60%, 80%, 100%` とし、前回実行で最も時間がかかったスレッドは必ず `100%` に残す。
- 他スレッドの MBA レベルは、前回実行の経過時間、前回の MBA %, PMU の LLC 参照/ミスを使い、最遅スレッドの時間に揃えるように決める。
- 非最遅スレッドの制限は、最低レベルの単純なクリップではなく、計算した MBA budget を全体的に緩めて低い MBA レベルへ寄りすぎないようにする。
- MBA target の再計算と resctrl group 振り分けは、CSV 表示と同じ `AMR_OMPT_EXEC_INTERVAL` ごとに行い、それ以外の実行では前回 target を維持する。
- PMU カウンタが取得できない環境では MBA 実行を継続しない。
- `AMR_RESCTRL_MBA_STRICTNESS` で PMU 圧力ペナルティの強さを調整できる。既定値は `1.15`。

## ビルド

```bash
bash build_amr.sh
```
