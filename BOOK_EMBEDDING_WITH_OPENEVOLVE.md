# Book Embedding with OpenEvolve (Cerebras gpt-oss-120b)

このリポジトリには、OpenEvolveで「ページ数最小化」のbook-embedding問題を進化させる最小雛形が入っています。

## 1. 前提

- Python 3.10+
- OpenEvolve
- Cerebras API Key（`CEREBRAS_API_KEY`）

## 2. セットアップ

```bash
pip install openevolve cerebras-cloud-sdk
export CEREBRAS_API_KEY="your-secret"
```

OpenAI互換エンドポイントではなく、`cerebras.cloud.sdk` を直接使う場合は
`examples/book_embedding/cerebras_sdk_client.py` の `generate_code_suggestion()` を利用してください。

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
- `examples/book_embedding/cerebras_sdk_client.py`  
  `cerebras.cloud.sdk` で直接モデル呼び出しを行うユーティリティ。
- `examples/book_embedding/config.yaml`  
  `cerebras_sdk` + `gpt-oss-120b` 設定。
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
