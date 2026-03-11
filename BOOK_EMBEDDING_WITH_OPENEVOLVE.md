# Book Embedding with OpenEvolve (Mistral AI Studio)

このリポジトリには、OpenEvolveで「ページ数最小化」のbook-embedding問題を進化させる最小雛形が入っています。

## 1. 前提

- Python 3.10+
- OpenEvolve
- Mistral API Key（`MISTRAL_API_KEY`）

## 2. セットアップ

```bash
pip install openevolve
export MISTRAL_API_KEY="your-secret"
```

`examples/book_embedding/mistral_ai_studio_client.py` の `generate_code_suggestion()` は
Mistral AI Studio API（`/v1/chat/completions`）を直接呼び出します。

このクライアントには **RPS=1 制約** を守るため、リクエスト間隔を最低1秒空けるレート制御を実装しています。

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
- `examples/book_embedding/mistral_ai_studio_client.py`  
  Mistral AI Studio APIを直接呼び出すユーティリティ（RPS=1レート制御付き）。
- `examples/book_embedding/config.yaml`  
  `mistral_ai_studio` + `mistral-large-latest` 設定。
- `examples/book_embedding/instances/*.json`  
  最小サンプルインスタンス。

## 5. OpenEvolve本体側の補足

OpenEvolve本体で `llm.provider` の値を固定で判定している場合は、
`mistral_ai_studio` を許可する変更が必要です。

- 例: provider名の許可リスト追加
- 例: Mistral AI Studio用の呼び出しクライアントを既存provider分岐へ接続

本リポジトリ側では、その前提に合わせて `config.yaml` の provider名を `mistral_ai_studio` に変更しています。

## 6. 評価方針

- 制約: 同一ページ内の辺が交差しないこと
- 主目的: ページ数最小化
- スコア:
  - 実行可能解（交差0）: `1000 - num_pages`
  - 非実行可能解: `1 / (1 + violations + num_pages)`

## 7. カスタマイズ

- 実問題用のグラフを `examples/book_embedding/instances/` にJSONで追加
- `config.yaml` の `max_iterations`, `population_size`, `num_islands` を拡張
- `prompt.system_message` に問題固有のヒント（禁則・局所探索戦略）を追記

## 8. 多様インスタンスの自動生成

以下を実行すると、`known_optima.json` の主要キー群（path/cycle/star/tree/K_n/K_{p,q}/grid/wheel/prism/random）を実体化し、
さらにseed固定のランダムグラフを追加して、合計約100件の評価インスタンスを再生成できます。

```bash
python tools/generate_instances.py
```

- 出力先: `examples/book_embedding/instances/*.json`
- 付帯情報: `examples/book_embedding/instances_manifest.csv`（family/頂点数/辺数/密度/生成元）
