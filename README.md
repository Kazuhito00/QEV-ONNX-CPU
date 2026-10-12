[[Japanese](README.md)/[English](README_EN.md)]

# QEV-ONNX-CPU

[`ken-jo/qev`](https://huggingface.co/ken-jo/qev)（QEV 0.2.1。構造化判定モデル。テキストと画像 1 枚）を、onnxruntime と numpy だけで推論する最小依存の推論実装です。 ※GPU推論も可（ただし、検証した GPU では速くなりませんでした。後述）<br>
公式の推論コード（PyPI の `qev`。モデルのコードは `veyra` パッケージ）が要求する torch・transformers は使いません。

# Features
以下の特徴があります。
- 最小依存：実行時に必要なパッケージは onnxruntime・numpy・pillow の3つだけ（pillow は画像を使うときだけ読み込みます）
- 動かすのに必要なスクリプトは `download_model.py`（モデル取得）と `demo_inference_text.py`（推論）の2つ。画像は `demo_inference_image.py`、カメラは `demo_inference_webcam.py`
- 判定は参照実装（QEV 公式の実行系、PyTorch fp32 CPU）と一致（検証済み。詳細は「Verification」）
- 1問が1回の順伝播で決まる（生成は走らない）。質問は3種類：`noul`（はい・いいえ）、`choice`（2〜16個から1つ）、`score`（2〜16段階の順序尺度）。どの答えにも確率・確信度・棄権（`abstained`）の印が付く
- 画像対応（**1リクエストにつき1枚まで**。QEV の規則です）
- 前処理（Qwen byte-level BPE、プロンプトの組み立て、3次元 M-RoPE、画像のパッチ化と視覚エンコーダの入力）を純Pythonで移植し、参照実装との突き合わせ検証を同梱
- CPU のほか、GPU（CUDA または DirectML）でも動かせる（`--device cuda`、`--gpu`。CPU への静かなフォールバックを防ぐ `--require-gpu` 付き）

# Purpose of This Repository
以下の検証を目的としています。
- QEV（Qwen3.5-2B ＋ LoRA ＋ 読み出し・条件・結合ヘッド）を、実行時依存3つだけでどこまで動かせるか
- 参照実装の前処理を純Pythonに移植して、同じ入力が作れるか
- 画像つきの判定が、ONNX 化した視覚エンコーダで元の実装と同じ答えになるか

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
pillow       10.0 or later   （画像を使う場合のみ読み込まれます）
```
実行に必要なのはこれだけです。モデルの取得も標準ライブラリのみで動きます。<br>
**メモリは、約 8 GB 以上の空きが必要です**（重みは fp32 で 7.5 GB。画像を使うと視覚エンコーダの 1.3 GB が加わります。ピーク RAM は計測していません）。<br>
環境構築は[uv](https://docs.astral.sh/uv/)を使います。<br>
オプションのグループがあります。
- `test`：`tests/` 用の pytest（モデルは要りません）
- `webcam`：`demo_inference_webcam.py` 用の OpenCV
- `oracle`：参照実装との突き合わせ検証用に、QEV 公式のパッケージ `qev`（Python 3.12 のみ。torch・transformers を引きます）
- `export`：ONNX を自分で書き出すときの torch・transformers など（実行時には使いません。`oracle` とは同居できません）
- extra `gpu`：`onnxruntime-gpu`（後述の「GPUで実行する」）

# Installation

```bash
git clone https://github.com/Kazuhito00/QEV-ONNX-CPU
cd QEV-ONNX-CPU

# 実行時の依存（onnxruntime・numpy・pillow）のみインストール
uv sync
```
`uv sync` は、このリポジトリの `qev_decide` パッケージも環境に入れます。`uv pip install git+https://github.com/Kazuhito00/QEV-ONNX-CPU` でも `import qev_decide` できます（モデルのファイルは別に取得します）。<br>
**PyPI の `qev` は、他のリポジトリの環境に入れないでください**（OmniJev の `qev` と import 名が同じです。この実装は `qev` を使いません。`oracle` グループは検証専用です）。

# Download Model
QEV には公開されている ONNX がありません。書き出した ONNX を[Releases](https://github.com/Kazuhito00/QEV-ONNX-CPU/releases)に置く前提です（リポジトリには含めていません。Release の作り方は `release/make_release.py` と下の「Release の中身」。**現時点では Release は未作成です**）。<br>
以下のスクリプトで `models/onnx/` に取得します。標準ライブラリのHTTPSのみで動き、gitもgit-lfsも不要です。
```bash
uv run download_model.py
```
- **引数なしでは、Release の全ファイル（L=256・384・512・1024・1280・2048 の全グラフ、視覚エンコーダ、ヘッド、トークナイザ、ライセンス文。25 個のアセット、約 9.9 GB）を取得します**
- 取得元は、既定で `Kazuhito00/QEV-ONNX-CPU` の最新の Release です（`--repo OWNER/NAME` か、環境変数 `QEV_ONNX_REPO` で別のリポジトリにできます。優先順位は 引数 > 環境変数 > 既定）
- 取得したファイルは、Release の `manifest.json` に書かれた SHA-256 と照合します（不一致ならファイルを削除して停止）
- GitHub の制限（1ファイル 2 GiB 未満）で分割されたファイル（共有の重み `torso_weights.data`、7.5 GB を 4 分割）は、取得後に結合して照合します（結合の間は、分割ファイルと結合後のファイルで、一時的に約 7.5 GB 余分にディスクを使います）
- 取得済みなら何もせず、中断した場合は同じコマンドで続きから再開します

```bash
uv run download_model.py --list                                          # ファイル一覧
uv run download_model.py --exclude "vision_encoder*"                     # テキスト専用（視覚エンコーダを取らない。約 1.3 GB 少ない）
uv run download_model.py --seq-len 384                                   # 既定の L=384 のグラフだけ（共有の重み・視覚エンコーダは付いてきます）
uv run download_model.py --tag v0.0.0                                    # タグを指定（既定は latest）
```

Release には L=256・384（既定）・512・1024・1280・2048 の6つのグラフが入っています。**重み（`torso_weights.data`、7.5 GB）は6つで共有しているので、グラフを1つ増やしても約 0.16〜0.18 GB しか増えません。** そのため `--seq-len 384` でも重みは 7.5 GB 取得します（節約できるのはグラフ 5 つ分の約 0.85 GB だけです）。`manifest.json` には、`.onnx` ごとに「必要なデータファイル」が書かれているので、`--seq-len` や `--exclude` で、グラフだけが残って重みが無い、ということは起きません。

自分で書き出す場合（元の checkpoint を取得して ONNX にします。torch が要り、書き出しはメモリ・時間とも重い作業です。詳細は下の「Building the ONNX files」）。

### Release の中身
`uv run release/make_release.py --dry-run` で一覧を出せます（ハッシュは計算せず、何も書きません）。`release/make_release.py` が `models/` から `dist/release/` に作ります（アップロードはしません）。1 ファイル 2 GiB 未満の制約のため、2 GiB を超えるのは共有の重みだけで、これを 1.9 GiB ごとに 4 分割します。

| アセット名 | サイズ |
|---|---:|
| `onnx__torso_L256.onnx` / `_L384` / `_L512` / `_L1024` / `_L1280` / `_L2048` | 161.2 / 162.6 / 164.0 / 169.4 / 172.2 / 180.4 MB |
| `onnx__torso_weights.data.part000` 〜 `part002`（共有の重み 7,527,333,888 バイトを分割） | 各 2,040,109,465 バイト（1.9 GiB） |
| `onnx__torso_weights.data.part003` | 1,407,005,493 バイト |
| `onnx__vision_encoder.onnx` / `onnx__vision_encoder.onnx.data`（1 アセット、2 GiB 未満） | 5.5 MB / 1,325,727,744 バイト |
| `onnx__head.npz` / `onnx__tokenizer.json` | 0.7 MB / 12.8 MB |
| `onnx__qev_config.json` / `onnx__preprocessor_config.json` / `onnx__graph_meta_L*.json`（6 個） | 各 1 KB 未満 |
| `onnx__LICENSE-QEV` / `onnx__NOTICE-QEV` / `onnx__LICENSE-QWEN` | 11 KB / 3.5 KB / 11 KB |
| `manifest.json` / `SHA256SUMS` | 小 |

合計は約 9.9 GB（25 アセット + `manifest.json` + `SHA256SUMS`）です。

# Usage

### 実行例(Python)
```bash
uv run demo_inference_text.py
uv run demo_inference_image.py
```
テキストのデモは、組み込みの例（カスタマーサポートの問い合わせに「怒っているか」「どのチームか」「どれくらい緊急か」を聞く）に答えます。自分の入力は次のように渡します。
```bash
uv run demo_inference_text.py --state "My card was charged twice." \
    --choice "team=billing,tech" --noul "angry=The customer is angry" --score "urgency=low,medium,high"
```
画像のデモは、既定で `artifacts/images/bars.png`（スクリプトで描いた棒グラフ）を使います。自分の画像を使うときは、画像のパスと質問を渡します（**1枚だけ**）。
```bash
uv run demo_inference_image.py shot.png --choice "status=paid,pending,failed" --noul "paid=Is it paid?"
```

```python
from qev_decide import Decider, choice, noul, score

decider = Decider("models/onnx")          # 既定: L=384（seq_len=512 などで変えられます）。環境変数 QEV_MODEL_DIR でも指定できます

decider.decide("My card was charged twice for the same order.", {
    "team": choice("Which team?", {"billing": "charges, refunds", "tech": "bugs"}),
    "angry": noul("The customer is angry."),
    "urgency": score("How urgent?", ["low", "medium", "high", "critical"]),
})
```
答えは QEV 自身の形式の辞書です：`probabilities`、`confidence`、`abstained`、そして `choice`（選んだキー）／`noul`（はいの確率）／`score`（期待値。`legend` に各段階の説明）。

**質問ごとに1回の順伝播です。** `choice` の選択肢は、QEV の仕様で**説明文の順に並べ替えられて**から文字（A、B、…）が振られるので、モデルが見る文字は呼び出し側の順序の順列になります（結果はキーで返ります）。同じ説明文を2つの選択肢に付けることはできません。

画像は同じリクエストに混ぜます（1枚）。
```python
decider.decide("Look at the screenshot.", {...}, images=["shot.png"])
```
`images` には、ファイルのパス（20 MiB 以下）、バイト列、Pillow の画像を渡せます（PNG・JPEG・WebP・GIF。40 MP まで）。QEV の画像処理は 65,536〜262,144 ピクセル・32 px 単位なので、小さい画像は**拡大**されます（224×224 は 256×256 になり、視覚トークンは 64）。透過 PNG は、アルファを捨てるだけです（背景色との合成はしません。QEV と同じ）。

### カメラで「人が映っているか」を判定する
OpenCV でカメラの映像を取得し、最新のフレームについて「人が映っているか」を判定して、結果を映像に重ねて表示します。
```bash
uv run --group webcam demo_inference_webcam.py                        # カメラ 0
uv run --group webcam demo_inference_webcam.py --camera 1 --mirror
uv run --group webcam demo_inference_webcam.py --question "Is the person wearing glasses?"
uv run --group webcam demo_inference_webcam.py --camera artifacts/images/bars.png   # カメラの代わりに画像・動画ファイル
```
- 判定は別スレッドで動かし、映像の表示は止めません。判定が出るたびに、上部の帯（緑＝はい／赤＝いいえ）と「はい」の確率が更新されます
- 質問は `--question`（はい・いいえの質問）で変えられます。モデルに渡す画像は長辺 224 px に縮小します（`--size`）
- `q` または `Esc` で終了します。OpenCV は、このデモだけの依存です
- カメラでの実機確認はしていません（未確認）。画像ファイルを渡して `--no-window --max-decisions 1` で 1 判定を出すところまでを、このリポジトリで確認しています

### 質問の種類と上限
| 種類 | 作り方 | 答え |
|---|---|---|
| `choice` | `choice(指示, {名前: 説明})`（2〜16個） | 選択肢から1つ（`choice`、`confidence`、`probabilities`） |
| `score` | `score(指示, [最低, ..., 最高])`（2〜16段階） | 順序尺度の期待値（`score`、`legend`、`confidence`）。先頭が最低 |
| `noul` | `noul("文")` | 文が成り立つか（`noul` = はいの確率）。`true` / `false` の説明は変えられます |

- 1リクエストに質問は1〜4個、画像は**1枚まで**、全質問の合計トークンは **2,048 まで**です（超えると `ValueError`。呼び出し側で入力を短くしてください）
- `state` には文字列を渡します（10 万文字まで）。画像だけで、文字が空でもかまいません
- 確率は QEV の較正（質問の種類ごとの温度と棄権のしきい値）を通した値です。この較正は、元の `manifest.json` に「親の方針を引き継いだもので、再フィットまでは開発用」と書かれています（下の「License」も参照）

### GPUで実行する（任意）
既定は CPU です。GPU（CUDA）で動かすときは、`onnxruntime` の代わりに `onnxruntime-gpu` を入れます。**この2つは同じ `onnxruntime/` フォルダに書き込むため、同居できません。** extra `gpu` を使うときは、入れたあとに入れ替えます。
```bash
uv sync --extra gpu                                              # onnxruntime-gpu も入る（CPU 版と同居した状態）
uv pip install --force-reinstall --no-deps onnxruntime-gpu       # 同じフォルダへ GPU 版を上書きして揃える
uv run --no-sync demo_inference_text.py --device cuda            # 以後は --no-sync で動かす（素の uv run は CPU 版に戻す）
```
- **要件**：CUDA 13 と cuDNN 9 の `bin` フォルダを PATH に入れます（`onnxruntime-gpu` は CUDA の DLL を同梱せず PATH から探します。入れたあとは、ターミナルを開き直します）。Windows の例です（バージョンは環境に合わせてください）。
  ```text
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64
  C:\Program Files\NVIDIA\CUDNN\v9.20\bin\13.2\x64
  ```
  DirectML（Windows。CUDA は不要）は、`onnxruntime-directml` を入れて `--device auto` で使います（このリポジトリでは動作を確認していません）
- **選び方**：`--device cpu`（既定）／`--device cuda`（CUDA、だめなら CPU）／`--device auto`（CUDA か DirectML があればそれ、なければ CPU）。`--gpu` は `--device auto` と同じです。`demo_inference_text.py`、`demo_inference_image.py`、`demo_inference_webcam.py`、`verify/bench.py` で使えます
- **静かな CPU へのフォールバックを防ぎます**：実際に使われたプロバイダを、デモの最終行（`provider: CUDAExecutionProvider` のように、最後に使ったグラフのセッションが実際に得たもの）と `verify/bench.py` の出力に表示します。GPU を頼んだのにロードできなかった（DLL が見つからない、VRAM 不足など）ときは、理由つきの警告を出して CPU で動きます。**`--require-gpu` を付けると、警告ではなくエラーで止まります**（CUDA を使えるビルドでなければ、`--device cuda` でも `--device auto` でも、モデルを読む前に止まります）
```python
from qev_decide import Decider, select_providers

decider = Decider("models/onnx", providers=select_providers("cuda"), require_gpu=True)
print(decider.provider)   # 実際に使われたプロバイダ
```
- **VRAM の目安**：胴体グラフの fp32 の重みだけで約 7.5 GB、視覚エンコーダが +1.3 GB です。VRAM 4 GB の GPU には載りません。十分な VRAM がある GPU のための任意機能です（そのような GPU での速度は未検証です）
- GPU での確率は、CPU と完全には一致しません（演算の順序の違い。下の測定では、最大の差は 3.0e-04）。参照実装との一致検証は CPU のみです

**GPU で速くなる、とは言えません。** 手元の測定では、次のとおりでした。
- 測定環境：RTX 3050 Ti Laptop 4 GB、CUDA 13.4、cuDNN 9.20、onnxruntime-gpu 1.31、画像は 224 px、CPU は 8 スレッド（作業時の測定。`eval_common/gpu_bench`、3 問×最大 10 周の中央値）
- CUDA セッションは作れました（全グラフが `CUDAExecutionProvider`。ピーク VRAM は 3,838〜3,892 MiB で頭打ち）が、1 問あたりの時間は、テキスト：CPU 4.5 秒に対し CUDA 5.4 秒、画像：CPU 5.0 秒に対し CUDA 5.9 秒で、**CPU より遅く**（約 0.8 倍）、GPU の利点はありませんでした。判定は CPU と一致（3/3、確率の差は最大 3.0e-04）です
- 重みが約 7.5 GB で 4 GB の GPU に載らないのに、例外も警告もなく CUDA セッションができました。理由は未検証です（作業時の推測は、Windows のドライバが GPU 専用メモリを超える分を共有メモリへ退避している、というもので、切り分けはしていません）。「CUDA セッションができた」ことは、速くなることを意味しません
- 作業ソースの README は、遅い理由を、約 9.2 万の小さなノードによる起動待ち（launch-bound）と説明していますが、このリポジトリでは検証していません
- 比較のため：同じ GPU の別モデルでは、JEVision・OneJev 0.8B などは CPU の 10 倍以上速い測定がありますが（このリポジトリのモデルとは無関係です）、重みが VRAM に収まらない 2B・4B 級のモデルでは速くなっていません

# 系列長は固定です（重要）
モデルの linear attention が 64 トークンのチャンクを Python のループで回すため、**エクスポート時に展開されて固定回数になります**。動的な系列長のグラフは作れません。

そのため固定長 L（64 の倍数）で書き出し、推論時に L まで**右パディング**します。読み出し位置はすべてパディングより前で、モデルは causal なので結果は変わりません（`verify/check_padding.py`）。**コストは実トークン数ではなく L に比例します。**

L=256、384、512、1024、1280、2048 の6つを Release に入れてあり、**既定は L=384** です（`Decider(..., seq_len=384)`、デモと `verify/bench.py` は `--seq-len`。極端に短い問い合わせには 256 を指定できます）。

- `seq_len` 以上で最小のグラフを最初に読みます
- リクエストが収まらないときは、**収まる最小のバケット**に自動で切り替えます（古いグラフを解放してから読むので、7.5 GB の胴体グラフがメモリに2つ載ることはありません）。**切り替えるたびにグラフの読み込み（作業時の記録で 45〜75 秒）が入る**ので、長さの違う問い合わせは長さ順に並べてください
- 最大の L=2048 を超えると `ValueError` です（QEV 自身の上限も、リクエスト全体で 2,048 トークンです）
- 6つのグラフは**重みのファイルを共有**します（`torso_weights.data`。バイト列が同一のテンソルを同じ位置から読むだけなので、数値は変わりません。書き出し時にテンソルごとにバイト単位で照合しています）。**`torso_weights.data` を消さないでください**
- 画像を使うと、プロンプトが長くなります（224 px の画像 1 枚で約 190〜230 トークン）

**速度の目安（根拠のある測定だけを載せます）**：

| 測定 | 値 | 出典 |
|---|---|---|
| CPU（8 スレッド）、画像つきの 1 問の中央値（Wikipedia 画像ベンチの Commons 1268 問、較正の合ったブロックのみ） | 4.7 秒 | `wikipedia-vision-decision-bench` の「1 問あたりの時間」表（移行前の測定） |
| CPU（8 スレッド）、1 問の中央値（テキスト／画像、RTX 3050 Ti Laptop の PC。GPU 比較の CPU 側） | 4.5 秒／5.0 秒 | `eval_common/gpu_bench`（作業時の測定） |

グラフごとの 1 問の時間（L 別）は、較正のとれた静かなマシンでの測定がまだ無いため載せません。メモリのピーク（RAM）も未計測です。ディスクから重みを読み込む最初の1回は、特に遅くなります（このリポジトリのデモは、グラフの読み込みを含む最初の1問に 96 秒かかりました。速度の測定ではありません）。測るときは、ほかの重い処理を止めて `uv run --no-sync verify/bench.py --seq-len N` を使ってください。

### 精度の目安
このリポジトリで新しく測った精度はありません。次は他の測定の引用です。Wikipedia 画像ベンチ（`wikipedia-vision-decision-bench`。8 スレッド・224px）での、この ONNX 実装の正答率（%）：

| Commons 英 | Commons 日 | synthetic 英 | synthetic 日 | reasoning 英 | reasoning 日 |
|---|---|---|---|---|---|
| 96.7 | 93.8 | 94.0 | 91.1 | 52.4（212 問） | 50.9（212 問） |

画像つき（Commons 1268・synthetic 382・reasoning 225）の値で、ベンチ側の「移行前の測定」です。QEV は 1 枚しか受け付けないため、reasoning は 2 枚の 13 問を除いた 212 問です。reasoning は選択肢が少なく当てずっぽうでも当たりやすい形式を含むため、偶然の正答率を意識して読んでください。元のモデルカード（[`ken-jo/qev`](https://huggingface.co/ken-jo/qev)）にも、独自の評価結果が載っています（PyTorch 実装での値です）。

# Verification
```bash
uv run verify/run_all.py --light                         # この環境の依存だけで動き、モデルの重みが要らないゲートだけ（check_prompt と run_blocked）
uv run verify/run_all.py                                 # + モデルを読むゲート（verify、check_padding。RAM が約 8 GB 以上要ります）
uv run verify/run_all.py --oracle --export               # + QEV 公式パッケージ・torch が要るゲート（重い。下の表）
uv run --group test pytest                               # 単体テスト（モデル不要）
```
`verify/run_all.py` は、各ゲートを必要な依存グループで `uv run` します。`oracle` と `export` は、torch を入れて環境を入れ替える重いグループなので、`--oracle`・`--export` を付けたときだけ動き、`--light` は決して入れません（同居もできません）。

| ゲート | 内容 | 必要なもの |
|---|---|---|
| check_prompt（と run_blocked） | プロンプト文字列・`input_ids`・条件位置が、参照実装（QEV 公式の実行系）と完全一致 | `models/onnx/tokenizer.json` |
| verify | 判定・確率・棄権の印・隠れ状態が参照実装（fp32）と一致（テキストと画像） | モデル |
| check_padding | 右パディングの量が読み出し位置の隠れ状態を動かさないこと（全グラフで確認） | モデル |
| check_vision_inputs | 画像のピクセル・位置 id が QEV の画像処理（transformers）と一致 | `--group export`（torch・transformers） |
| check_tokenizer_fuzz | プロンプト生成を veyra と HF のプロセッサに対してファズ | `--group oracle` と `QEV_HOME` |
| check_bpe_fuzz / check_unicode_classes | BPE を transformers・tokenizers と 15 万テキスト／全コードポイントで突合 | `--group oracle` と `QEV_HOME` |
| run_blocked | `tokenizers`・`transformers`・`torch` を import できなくして、ゲートが動くこと（実行時にそれらが要らないことの確認） | — |

`artifacts/golden.json`・`golden_hidden.npz`・`golden_image.json`・`golden_image.npz` は、参照実装（QEV 公式の実行系、PyTorch fp32、CPU）を実際に動かして作った正解データです。同じ重みを両者が走らせるので、差が出たら移植のバグと断定できます。`artifacts/images/` の3枚（`red_circle.png`・`green_triangle.png`・`bars.png`）は、スクリプトで描いた画像です。**正解データには、Wikipedia 由来の実写画像を使った6問（`img-wiki*`）も含まれますが、その画像はこのリポジトリにありません**。`verify/verify.py` は、画像が無い問を飛ばします（`skipped`）。

このリポジトリで `uv run verify/verify.py --threads 6` を再実行した結果（2026-10-11、CPU、fp32）：画像が手元にある **22 問**（テキスト 19 問と、スクリプトで描いた画像 3 枚の問）で、**トークン id の不一致 0、判定の不一致 0、棄権の印の不一致 0**。最大の差は、隠れ状態 1.14e-04、条件ロジット 2.92e-05、選択肢ロジット 6.94e-05、確率 1.46e-06、視覚エンコーダ出力 1.50e-04 でした。`check_prompt` は 28/28 で完全一致です（実写画像の問も、トークン id は文字列だけから作れるため確認できます）。作業時（元の作業ツリー、実写画像 6 問を含む 28 問）の記録では、判定・棄権の不一致は 0 のままで、最大差は実写画像の問で大きくなりました（隠れ状態 1.54e-01、確率 4.84e-03、視覚エンコーダ出力 3.43e-01。画像の前処理が QEV の torch 版と一部のピクセルで 1〜2 階調ずれるため）。`check_padding`、`check_vision_inputs`、`check_tokenizer_fuzz`、`check_bpe_fuzz`、`check_unicode_classes` は、このリポジトリでは再実行していません（`oracle` / `export` グループの torch・transformers と元の checkpoint が要るため。作業時の元の作業ツリーでは通っています）。`tests/` の単体テストは、このリポジトリで通っています。

# Building the ONNX files
自分で書き出すときの手順です（torch が要り、メモリも時間も重い作業です。元の checkpoint は QEV の `qev download` が、リビジョン固定・SHA-256 照合つきで取得します）。
```bash
uv sync --group oracle                                          # QEV 公式のパッケージ（Python 3.12）
QEV_HOME=models/qev_home uv run --group oracle qev download     # 固定されたチェックポイントと Qwen3.5-2B（約 4.6 GB）
QEV_HOME=models/qev_home uv run --group oracle export/make_golden.py        # 正解データ（QEV 公式の実行系で作る）
QEV_HOME=models/qev_home uv run --group oracle export/make_golden_image.py
bash export/export_all.sh 256 384 512 1024 1280 2048            # 固定長のグラフ（1本ずつ。--group export。重い）
uv run --group export export/share_weights.py --apply           # 重みのファイルを 1 つに寄せる（バイト列が同一のテンソルだけ）
uv run --group export export/export_vision.py                   # 視覚エンコーダ
```
`export_onnx.py` は `models/onnx/head.npz`、`tokenizer.json`、`preprocessor_config.json`、`graph_meta_L<N>.json` も書きます。`oracle` と `export` は同居できません（uv が片方に入れ替えます）。このリポジトリでは、`export/` のスクリプトを uv のグループ構成に直した後で再実行していません（作業時の手順を整理したものです）。

# Limitations
- 画像は **1リクエストにつき1枚まで**、質問は 1〜4 個、`choice`・`score` は 2〜16 個、全体で 2,048 トークンまでです（QEV の規則です）
- 系列長は固定です。既定は L=384 で、収まらなければ大きいバケットに切り替え、その読み込みに 1 分前後かかります
- fp32 のみです（fp16 は作っていません）。動画は未対応です
- メモリを多く使います（重みだけで 7.5 GB。最小でも約 8 GB の空きが要ります）
- 画像の前処理は、QEV の（torch の）画像処理と、一部のピクセルで 1〜2 階調ずれます（Pillow の bicubic のため）。実写画像で、視覚エンコーダの出力と隠れ状態の差が大きくなる原因です（判定は一致しています）
- 確率の較正は、QEV の `manifest.json` で「親の方針を引き継いだもので、再フィットまでは開発用」とされています（`release_allowed: false`）。この実装は、その値をそのまま使います
- GPU では確率が CPU と完全には一致せず、4 GB の GPU では速くなりません

# Project Structure

```text
README.md                # README（日本語）
README_EN.md             # README（英語）
LICENSE                  # Apache-2.0
pyproject.toml           # 依存定義（実行時は onnxruntime・numpy・pillow。グループ: test / webcam / oracle / export、extra: gpu）
uv.lock                  # 依存の固定
download_model.py        # モデル取得（GitHub Releases、SHA-256照合、分割ファイルの結合）
demo_inference_text.py   # テキストの推論デモ
demo_inference_image.py  # 画像の推論デモ（1 枚）
demo_inference_webcam.py # カメラで「人が映っているか」を判定（OpenCV）
qev_decide/              # 推論実装（軽い依存のみ。export/ を import しない）
  runtime.py             #   ORT セッション + 右パディング + 読み出し + バケット切り替え
  _ort.py                #   セッションの生成・CPU/GPU の選択（--device / --gpu / --require-gpu）
  tokenizer.py           #   Qwen byte-level BPE の純Python実装
  protocol.py            #   リクエストの検証・プロンプトの組み立て・ヘッド・温度・棄権の計算
  image.py               #   Pillow デコード + smart_resize + パッチ化 + 視覚エンコーダの入力 + 3次元 M-RoPE
  _regex_translate.py    #   Rust 方言の正規表現を Python の re に直す
  _uniprops.py           #   生成物: Unicode範囲表
export/                  # 重い依存（torch / transformers / onnx / qev）
  export_onnx.py         #   胴体（LoRA 畳み込み済み）を固定長の ONNX に
  export_vision.py       #   視覚エンコーダを ONNX に（vision_export_patch.py で単一画像版に差し替え）
  share_weights.py       #   バケット間で重みのファイルを共有（同一バイト列のテンソルだけ）
  export_all.sh          #   複数の長さを順に書き出す
  make_golden.py         #   artifacts/golden.json を作る（cases.py の質問を QEV 公式の実行系で走らせる）
  make_golden_image.py   #   artifacts/golden_image.* を作る
  gen_uniprops.py        #   _uniprops.py を作る
release/
  make_release.py        # models/ から Release 用のファイルを作る（分割・SHA-256・manifest.json）。--dry-run で一覧だけ
  notes.md               # Release の本文
  licenses/              # Release に同梱するライセンス文（QEV、QEV の NOTICE、Qwen）
tests/                   # 単体テスト（モデル不要。プロバイダ選択のテストを含む）
verify/                  # 検証ゲート（run_all.pyで一括）、bench.py（1 問の時間の測定）
artifacts/               # golden*（検証の正解データ）、images/（デモとテスト用の画像）
models/                  # ONNX・トークナイザ・設定（gitignore）
```

# Reference
- [ken-jo/qev](https://huggingface.co/ken-jo/qev)（元のモデル。LoRA アダプタと各ヘッド。Apache-2.0）、[PyPI: qev 0.2.1](https://pypi.org/project/qev/)（元の推論コード。モデルのコードは `veyra` パッケージ。Apache-2.0）
- [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)（元のモデルのベース。Alibaba Cloud。Apache-2.0。リビジョン 15852e8c）

# License
QEV-ONNX-CPU is under [Apache-2.0 license](LICENSE).<br>
モデルの重みは、元のモデル（[ken-jo/qev](https://huggingface.co/ken-jo/qev)、モデルカードのライセンスは Apache-2.0。ベースの Qwen3.5-2B も Apache-2.0）のライセンスに従います。Release には、QEV のライセンス文と NOTICE（PyPI パッケージ `qev` 0.2.1 に同梱されていたもの）、Qwen のライセンス文を元のまま同梱しています（`release/licenses/`）。<br>
`qev_decide/` のコードは、QEV（Apache-2.0）の推論コード、Hugging Face Transformers（Apache-2.0）の Qwen2-VL の画像処理や Qwen3.5 の視覚位置の計算を、numpy で書き直した移植です。

**注意（再配布の前に確認してください）**：QEV の NOTICE には、学習に使われたデータセットの一覧があり、次のものが含まれます。
- SNLI（CC-BY-SA-4.0）：NOTICE は、「選んだ前提・仮説の組から作った判定用のデータは、継承される同一条件での共有（share-alike）の条件を保つ」と書いています。この条件が、学習済みの重み（とその変換物）に及ぶかどうかは、このリポジトリでは判断していません
- CIFAR-10：画像の転用評価にだけ使われ、画像は再配布されていません。NOTICE は「データセットカードのライセンスは不明で、再配布の許可を示すものではない」と書いています
- BANKING77（CC-BY-4.0）、LocalLLaMA/typed-decisions（Apache-2.0）、Beans・TrashNet（MIT）など

なお、同梱の NOTICE は `data-licenses/` 内の各データのライセンス文（Hugging Face の元リポジトリにあります）を参照していますが、それらは Release には含めていません。

また、QEV のチェックポイントの `manifest.json` には、`"release_allowed": false`（較正は「再フィットまでは開発用」）と書かれています。一方で、Hugging Face のモデルカードは、コードと適応重みを Apache-2.0 としています（2026-10-11 時点）。この食い違いの扱い（`release_allowed` の意味）は、確認できていません。

## 変換物であることについて
Release に含まれる ONNX ファイルは、公式には ONNX が公開されていない [QEV](https://huggingface.co/ken-jo/qev)（Apache-2.0。LoRA アダプタと各ヘッドだけのリポジトリ）を、このリポジトリの `export/` で ONNX 形式に変換したものです。具体的には、次の改変をしています。
- ベースモデル [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)（Alibaba Cloud。Apache-2.0）の重みに、QEV の LoRA アダプタ（rank 8、全 24 層。倍率 alpha / r = 2）を畳み込み、テキストのデコーダ（胴体）を fp32 の ONNX グラフとして書き出しています。元のリポジトリには、ベースの重みは含まれていません
- 系列長ごとの固定長グラフに分け、グラフ間で重みのファイルを共有しています
- ベースモデルの視覚エンコーダ（Qwen3.5-2B のもの。重みは無調整）を、単一画像用の ONNX として書き出しています
- 読み出し・条件・結合ヘッド（LayerNorm と射影と専門家の重み）は、`head.npz`（fp32）に変換し、numpy で計算します。温度と棄権のしきい値は `qev_config.json` に写しています

重みは fp32 のまま書き出しており、元の実装（PyTorch fp32）との判定の一致を確認しています（「Verification」）。元のモデルの提供元およびこの変換物の動作を、提供元は保証していません。
