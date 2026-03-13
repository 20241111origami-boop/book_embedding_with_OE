# Book Embedding with OpenEvolve (OpenRouter BYOK)

このリポジトリには、OpenEvolveで「ページ数最小化」のbook-embedding問題を進化させる最小雛形が入っています。

## 1. 前提

- Python 3.10+
- OpenEvolve
- OpenRouter API Key（`OPENROUTER_API_KEY`）

## 2. セットアップ

```bash
pip install openevolve
export OPENROUTER_API_KEY="your-secret"
```

`examples/book_embedding/mistral_ai_studio_client.py` の `generate_code_suggestion()` は
OpenRouter API（`/api/v1/chat/completions`）を直接呼び出します。

このクライアントには **RPS制御 + 429/5xx時のモデル・フェイルオーバー（best-effort）** を実装しています。
デフォルトの切り替え順は次の通りです。

1. `cerebras/gpt-oss-120b`
2. `groq/gpt-oss-120b`
3. `z-ai/glm-4.7-flash`
4. `google/gemini-3.1-flash-lite`
5. `openrouter/hunter-alpha`

必要なら `OPENROUTER_MODEL_CANDIDATES` で順序を上書きできます（カンマ区切り）。

### OpenRouter BYOK の重要な注意点

- BYOKキーを登録しているプロバイダーは、`provider.order` を指定しても優先されることがあります。
- デフォルトではBYOKが失敗した際に共有クレジット側へフォールバックし得ます。これを避けたい場合は `OPENROUTER_ALLOW_FALLBACKS=false` を設定してください。
- 同一プロバイダーに複数BYOKキーがある場合、使用順序は保証されません。
- `OPENROUTER_PARTITION=none` を使うと、BYOK登録プロバイダーをモデル横断で優先しやすくなります。
- 失敗時はOpenRouterダッシュボードの Activity -> View Raw Metadata で実際のルーティング先とエラー理由を確認してください。

### ルーティング関連の環境変数

- `OPENROUTER_MODEL_CANDIDATES` : モデル候補をカンマ区切りで指定
- `OPENROUTER_PROVIDER_ORDER` : OpenRouter `provider.order` を指定
- `OPENROUTER_ALLOW_FALLBACKS` : `true/false` で provider fallback 許可を切替
- `OPENROUTER_PROVIDER_SORT` : OpenRouter `provider.sort` を指定
- `OPENROUTER_PARTITION` : OpenRouter `provider.partition` を指定（例: `none`）


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
  OpenRouter APIを直接呼び出すユーティリティ（RPS制御 + 複数モデルフェイルオーバー）。
- `examples/book_embedding/config.yaml`  
  OpenRouter BYOK設定。
- `examples/book_embedding/instances/*.json`  
  最小サンプルインスタンス。

## 5. OpenEvolve本体側の補足

OpenEvolve本体で `llm.provider` の値を固定で判定している場合は、
`openrouter` または利用中のprovider名を許可する変更が必要です。

- 例: provider名の許可リスト追加
- 例: OpenRouter用の呼び出しクライアントを既存provider分岐へ接続

本リポジトリ側では、OpenRouter BYOKを使う前提で `config.yaml` のAPI設定をOpenRouter向けにしています。

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
