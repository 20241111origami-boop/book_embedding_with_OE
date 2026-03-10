# Book Embedding with OpenEvolve (Cerebras gpt-oss-120b)

このリポジトリには、OpenEvolveで「ページ数最小化」のbook-embedding問題を進化させる最小雛形が入っています。

## 1. 前提

- Python 3.10+
- OpenEvolve
- Cerebras API Key（`CEREBRAS_API_KEY`）

## 2. セットアップ

```bash
pip install openevolve
export CEREBRAS_API_KEY="your-secret"
# OpenEvolveはOPENAI_API_KEY環境変数を参照するため橋渡し
export OPENAI_API_KEY="$CEREBRAS_API_KEY"
```

## 3. 実行

```bash
python openevolve-run.py \
  examples/book_embedding/initial_program.py \
  examples/book_embedding/evaluator.py \
  --config examples/book_embedding/config.yaml \
  --iterations 120
```

## 4. ファイル構成

- `examples/book_embedding/initial_program.py`  
  進化対象。`solve_instance(graph)` が spine順序とページ割当を返します。
- `examples/book_embedding/evaluator.py`  
  各インスタンスで交差違反とページ数を測定し、`combined_score` を返します。
- `examples/book_embedding/config.yaml`  
  Cerebras (`https://api.cerebras.ai/v1`) + `gpt-oss-120b` 設定。
- `examples/book_embedding/instances/*.json`  
  最小サンプルインスタンス。

## 5. 評価方針

- 制約: 同一ページ内の辺が交差しないこと
- 主目的: ページ数最小化
- スコア:
  - 実行可能解（交差0）: `1000 - num_pages`
  - 非実行可能解: `1 / (1 + violations + num_pages)`

## 6. カスタマイズ

- 実問題用のグラフを `examples/book_embedding/instances/` にJSONで追加
- `config.yaml` の `max_iterations`, `population_size`, `num_islands` を拡張
- `prompt.system_message` に問題固有のヒント（禁則・局所探索戦略）を追記

