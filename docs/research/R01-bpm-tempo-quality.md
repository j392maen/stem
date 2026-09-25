# R01 拍・BPM・速度変更・分離品質の調査

調査日: 2026-09-25。調査担当（コード変更なし・インストールなし）。
凡例: **[確認]** は一次情報または手元で実測して確かめたこと。**[推測]** は裏付けが弱く、実装前に試す必要があること。

---

## 0. 結論（先に要点）

| 項目 | 推奨 | 予備 |
| --- | --- | --- |
| 拍・小節の頭の検出 | **beat_this**（CPJKU、MIT、ISMIR 2024）。元の曲（mixture）を入力にする | librosa `beat_track`（導入済み。ただし小節の頭は出ない。フォールバック用） |
| データの持ち方 | 拍の時刻の列（自動結果）＋ユーザーの**アンカー**（ワープマーカー）を別に保存。BPM は拍の間隔から区間ごとに計算 | — |
| ピッチも変わる速度変更 | Web Audio の `playbackRate` を全 stem に同じ時刻で設定（すぐ効く。追加なし） | — |
| ピッチを保つ速度変更 | **サーバーで事前変換**（導入済みの ffmpeg の `rubberband` フィルタ → Opus、速度ごとにキャッシュ）。1曲 8 stem で約 10 秒 | ブラウザ内の signalsmith-stretch（MIT、WASM）。PC で連続的に変えたいとき |
| サブボーカル | karaoke を **anvuew（BS）** に変え、frazer と平均。karaoke を元の曲に直接かける方式を比べる | Gabox karaoke v2 を加える |
| 「その他」の分割 | audio-separator には strings / synth / pad 用のモデルが**無い**。まずは信号処理（HPSS）で「伸びる音（パッド）」と「短い音（ヒット）」に分けるのを試す | MSST 形式の Mega 53 stems（VRAM 16GB 推奨で 8GB では要検証）、MVSEP のオンライン専用モデル（外部サービス） |

加えて、**今のパイプラインが「その他」にボーカルの残りを入れてしまう仕組み**を見つけた（D-1 参照）。

---

## A. 拍・ダウンビート（小節の頭）の検出

### A-1. 候補の比較

| 候補 | 精度（公開値） | 小節の頭 | テンポが途中で変わる曲 | Windows + Py3.12 + torch 2.11 | 処理時間の目安 | ライセンス |
| --- | --- | --- | --- | --- | --- | --- |
| **beat_this**（v1.1.0, 2026-04） | GTZAN: 拍 F1 89.1、小節の頭 F1 78.3（DBN なし）。DBN ありは F1 88.1 / 77.4 だが連続性指標（CMLt/AMLt）は上がる [確認: 論文 表2] | ○ | 強い。テンポや拍子を固定する後処理（DBN）を使わず、テンポ変動の大きいクラシックや拍子が変わる曲も学習している [確認: 論文 abstract] | ほぼそのまま動く見込み。必要なもの: `beat-this`（純 Python）、`torchaudio`（2.11 が最後の版で cu128 版あり）、einops・soxr・rotary-embedding-torch・tqdm（**すべて導入済み** [確認: .venv]）。C 拡張のビルド不要。重み 78MB（small は 8.1MB） | [推測] GPU で 1 曲数秒、CPU でも数十秒以内（2,000 万パラメータ、22kHz・50fps の mel を 30 秒単位で処理 [確認: 論文]） | コード・重みとも **MIT** [確認: README] |
| All-In-One（allin1 1.1.0, 2023-10） | Harmonix で拍・小節の頭・構成（サビ等）すべて当時の最高 [確認: 論文 abstract] | ○（＋曲の構成ラベル） | ○（分離した stem を入力に使う） | **難しい**。NATTEN が必須で Windows ではソースからビルド、madmom も GitHub から、demucs も要る [確認: README]。NATTEN の新版は API が変わり、allin1 は 2023 年から更新なし | RTX 4090 で 10 曲 33 分を 73 秒 [確認: README] | MIT [確認: README] |
| madmom（0.16.1 は 2018 年） | 以前の標準（DBN 付き） | ○ | DBN がテンポ範囲を制約。大きな変化に弱め | PyPI 版は Python 3.10 以上で import に失敗（`collections.MutableSequence`）。GitHub の main から入れる必要あり＋Cython ビルド [確認: 検索結果・issue] | CPU で十数秒程度 [推測] | コード BSD、**モデルは CC BY-NC-SA 4.0**（非商用）[確認: README] |
| BeatNet | 論文時点で上位 | ○ | 粒子フィルタ（リアルタイム向け）/DBN | madmom と pyaudio に依存。madmom 0.16.1 は Py3.10 以上・NumPy 1.24 以上で問題と明記 [確認: README] | — | CC BY 4.0 [確認: README] |
| librosa `beat_track`（**1.0.0 導入済み**） | 古典的手法。深層学習系より低い | **×**（拍のみ） | 既定では曲全体で1つのテンポを仮定。時間で変わるテンポを渡す拡張はある [確認: librosa 文書] | 動く [確認: 実行] | 手元で 184 秒の曲: 初回は numba のコンパイル込みで約 10 秒、2 回目 0.35 秒 [確認: 実測] | ISC |

手元での librosa の試行 [確認: 実測]: stems/3 の曲で、元の曲からは 143.6 BPM、drums stem からは 136.0 BPM と別の値が出た（librosa のテンポ候補の刻みで隣の値に揺れている）。小節の頭が取れないこともあり、主役にはしない。

### A-2. 分割済み stem を入力に使うと良くなるか

- 分離した stem を入力にする方式（Beat Transformer、All-In-One）は、分離しない場合より小節の頭の精度が上がると報告されている（Beat Transformer は TCN 比で小節の頭が最大 4 ポイント向上）[確認: Beat Transformer 論文 abstract]。ただしこれは**モデル自体が複数 stem を入力として学習**した場合の話。
- beat_this は元の曲（mixture）で学習している。drums stem だけを入れると、ドラムの無い区間（イントロ、バラード部）で拍を見失う。[推測]
- 分離後の drums stem でも音の立ち上がり時刻（onset）はほぼ保たれる、という最近の報告がある（アタックの形は崩れる）[確認: arXiv 2609.04224]。
- **推奨**: まず元の曲を beat_this に入れる。改善の余地があれば「元の曲」と「drums+bass」の2通りを走らせ、拍の確率を平均する（beat_this は `Audio2Frames` で拍の確率の列を出せる）。これは試してから決める [推測]。

### A-3. 推奨

- **beat_this（final0）を GPU で、分割ジョブの後処理として 1 回だけ実行**し、結果を保存する。分離モデルと同じく、使い終わったら GPU メモリを解放する。
- 導入に必要なこと（ユーザー確認）: `beat-this` と `torchaudio==2.11`（cu128）を gpu 用の optional 依存に追加。重み 78MB を初回にダウンロード。beat_this は `torchaudio` をモジュールの先頭で import するので省けない [確認: preprocessing.py]。音声の読み込みは soundfile で自前に行い `Audio2Beats` に配列を渡せば、torchaudio の読み込み機能（TorchCodec が必要）は使わずに済む。
- 予備: librosa（拍のみ。小節の頭は「最初の拍から4拍ごと」を仮に置き、ユーザーが直す）。
- 使わない: allin1（Windows で NATTEN のビルドが要る）、madmom 系（古く、モデルが非商用）。

---

## B. 可変テンポの表し方と手動補正

### B-1. データの持ち方（提案）

1. **自動結果**（再計算で上書き可）: `beats: [秒, …]`、`downbeat_flags` または各拍の「小節内の何拍目か」、`algorithm`（例 beat_this final0）、作成日時。
2. **ユーザーの補正**（自動結果とは別に保存し、再解析で消さない）: アンカー（ワープマーカー）の列 `[{time_sec, beat_index}]`、`beats_per_bar`（区間ごとに変えられるように `[{from_bar, beats_per_bar}]`）、`bar_offset`（どの拍を 1 小節目の頭にするか）。
3. **使うときの拍の列** = 自動結果をアンカーで補正したもの。アンカーとアンカーの間は拍を等間隔に置く（区分線形。Ableton のワープマーカー、Serato の Beat Warp Marker と同じ考え方）。
4. **BPM の表示**: 局所 BPM = 60 / 拍の間隔。そのままだと揺れるので、4〜8 拍の中央値で平滑化し、差が ±1.5% を超える所で区間を切る。画面には「今いる区間の BPM」を出し、区間の境目を波形の上に印で示す。
5. 保存先: 新しい表（例 `BEAT_GRID`: track_id, source=auto/manual, algorithm, beats_path もしくは beats_json, anchors_json, meter_json, updated_at）。4 分の曲で 130 BPM なら約 520 拍で JSON でも数 KB。

小節線の描画: 拍＝細い線、小節の頭＝太い線＋小節番号。拡大表示で拍の間隔が数 px より狭くなったら拍の線を省き、小節線だけにする。

### B-2. 補正 UI の定石（市販ソフト）

| ソフト | 可変テンポ | 主な補正操作 |
| --- | --- | --- |
| rekordbox | 解析モード「Dynamic」でテンポ変化の所に複数のマーカーを置く [確認: Lexicon の解説]。一方で最初の拍を動かすと全体がずれる等、局所的な直しが苦手との声 [確認: Pioneer DJ フォーラム] | 最初の拍の位置、グリッド全体のずらし、BPM の直接入力、タップ |
| Serato DJ | Beat Warp Marker（手動の赤い太線）で区間ごとにテンポを変える。Serato DJ 5.0 の Lucid Beatgrids で曲全体のテンポ変化を自動で追う [確認: Serato サポート] | Slip（BPM を変えずに全体をずらす）、Downbeat Marker（1 小節目の頭）[確認: Serato サポート] |
| Traktor Pro 4 | Flexible Beatgrid（2024-07）。マーカーを手で置いてテンポの変わり目を教える [確認: Digital DJ Tips] | マーカーの追加・移動 |
| Ableton Live | ワープマーカー。「Set 1.1.1 Here」で曲の頭を決め、「Warp From Here」で右側を自動解析し直す [確認: Ableton マニュアル] | マーカーをドラッグ、右側だけ再解析 |

stemapp で最低限そろえるもの（提案）:

- **1 小節目をここに**（再生位置の近くの拍を小節の頭にする。拍の位置は変えず番号だけずらす）。
- **×2 / ÷2**（倍・半分のテンポの誤り。拍を間に足す・1 つおきに間引く）。
- **拍子**（4/4、3/4、6/8 など。区間ごと）。
- **タップ**（再生しながらキーを叩く→叩いた時刻に最も近い拍にアンカーを置く、または叩いた間隔から BPM を出す）。
- **キューから計算**: キュー A と B の間の小節数 N を入れる → その区間の BPM = N × 拍子 × 60 / (B − A)。A と B をアンカーにし、間の拍を等間隔に置き直す（ユーザー要望3の「キューを打って計算」はこの形）。
- **ここから再解析**（Ableton の Warp From Here 相当）は後回しでよい。beat_this は曲全体を一度解析すれば済むので、区間ごとの再解析より「この区間の拍を等間隔にする」で足りることが多い [推測]。

### B-3. BPM を変える操作の意味（可変テンポの曲）

速度変更は「曲全体に同じ倍率 r を掛ける」とするのが自然（テンポの揺れはそのまま残る）。画面の BPM 入力は「今の区間の BPM を X にする」と解釈し、r = X / 今の区間の元の BPM とする。表示は各区間とも「元の BPM × r」。拍・小節線の位置は、再生位置が元の曲の時刻で管理されていればそのまま使える（C-1 参照）。

---

## C. 速度変更（全 stem 同期のまま）

### C-1. ピッチも変わる方式: `playbackRate`

- `AudioBufferSourceNode.playbackRate` は k-rate（128 サンプルごとに1回更新）の AudioParam。値を変えると速さと音の高さが一緒に変わり、ピッチ補正は無い。ループ位置（loopStart/loopEnd）は**音声データ上の時刻**のままで、速度の影響を受けない [確認: MDN]。
- 同期: 全 stem の source に `playbackRate.setValueAtTime(r, t)` を**同じ t（少し先の AudioContext 時刻）で**設定すれば、同じ描画単位で切り替わるので stem 同士はずれない。今の engine.js が start を同じ時刻に揃えているのと同じ理屈 [推測: 仕様上そうなるはず。実機で確認]。新しく作る source（シーク時など）にも同じ r を入れること。
- 再生位置の計算: 今の `position = startOffset + elapsed` を `startOffset + elapsed × r` にする。速度を変えた瞬間に `startOffset` を今の位置、`startCtxTime` を切り替え時刻に置き直す（`setLoop` と同じやり方）。ランプ（徐々に変える）を使うと位置の計算が積分になって面倒なので、段階的に切り替える。ループの折り返しは元の曲の時刻で判定すればよい。
- 手間が小さく、PC でも iPhone でもすぐ使える。

### C-2. ピッチを保つ方式

#### (a) ブラウザ内で処理する

| ライブラリ | 方式・品質 | ライセンス | 同梱 | 注意 |
| --- | --- | --- | --- | --- |
| **signalsmith-stretch**（Web 版） | スペクトル処理（位相ボコーダ系）。0.75〜1.5 倍向けに調整 [確認: README] | **MIT** [確認] | 公式の Web 版 `SignalsmithStretch.mjs`（WASM＋AudioWorklet）をファイルで同梱できる。npm 不要 [確認: web/release] | 音声データを `addBuffers` でノードに渡して再生するので、**PCM を二重に持つ**（PC で 700MB → 約 1.4GB）[推測]。`schedule({output, input, rate})` で出力時刻・入力位置・速度を指定できるので、全 stem のノードに同じ値を渡せば同期させられる。ライブ入力モードでは rate・input が無視されるので、「全 stem を混ぜてから 1 回だけ伸ばす」使い方はできない [確認: README] |
| SoundTouchJS（@soundtouchjs/audio-worklet） | WSOLA（波形の切り貼り）。位相ボコーダ版もある | MPL-2.0 [確認] | npm 前提の構成。ファイルを取り出して同梱は可能 [推測] | WSOLA は伸ばし方が音ごとに変わるため、stem ごとに別々に処理すると stem 間で数 ms のずれが出うる [推測] |
| rubberband-wasm 系 | Rubber Band（高品質） | **GPL**（個人利用なら問題になりにくいが、配布するなら要注意） | 非公式ビルド | 重い。8 stem 同時は厳しい [推測] |

- 8 stem 同時: 伸縮処理を stem の数だけ回す必要がある（stem の ON/OFF を後段の音量で切り替えるため）。PC（i7-14650HX）なら 8 本でも動く見込みだが、**iPhone では CPU と電池の負担が大きく、途切れやすい**と考えるべき [推測]。iOS Safari では画面回転などの操作で AudioWorklet の音が割れるという報告もある [確認: WebAudio issue #2632]。AudioWorklet 自体は iOS Safari 14.5 以上で使える [確認: caniuse]。
- 速度を変えるたびにすぐ反映できるのが利点。

#### (b) サーバーで事前に変換する

- **導入済みの ffmpeg 8.0（gyan.dev full）に `rubberband` フィルタが入っている** [確認: `ffmpeg -filters`、`--enable-librubberband`]。オプションは tempo / pitch / transients / window / formant 等。エンジンの選択肢は見当たらず、高品質版の R3（Finer）ではなく **R2 エンジン**を使うと思われる [推測: フィルタのオプション一覧に engine が無い]。
- **手元の実測** [確認]:

| 処理（184 秒の other stem、44.1kHz ステレオ） | 時間 |
| --- | --- |
| `rubberband=tempo=1.1` → WAV | 7.8 秒（約 23 倍速、1 スレッド） |
| 同 → Opus 128kbps | 7.9 秒 |
| `atempo=1.1`（WSOLA 系、軽い） | 0.3 秒 |
| `asetrate`+`aresample`（ピッチも変わる） | 0.2 秒 |

  CPU は 24 スレッドあるので、8 stem を並列にすれば**4 分の曲で 10 秒前後**の待ち時間と見込める [推測: 1 本の実測からの計算]。

- **stem 間の同期** [確認: 実測]: 同じ曲の drums / bass / other / lead_vocal の 30 秒を別々に rubberband（1.1 倍）で伸ばした。出力の長さは 4 本とも同じサンプル数。単純なリサンプル（時刻のずれが無い基準）と音の立ち上がりを比べると、ずれは 4 本とも 2.9〜4.4 ms で、**stem 同士の差は測定の刻み（約 1.5 ms）程度**。聞いて分かるずれではない。なお「伸ばした stem の合計」と「合計を伸ばしたもの」は波形としては一致しない（位相の処理が非線形のため）。音として問題になるかは聴いて確かめる。
- 容量: Opus 128kbps で 4 分 1 stem 約 3.8MB。stem 8 本で約 31MB／速度 1 つ。速度を 1% 刻みにし、使った速度だけ残す（古いものから消す）形なら問題にならない。STEM_RENDITION に `tempo_ratio` と `pitch_mode` の列を足して記録するのが素直。
- 高品質化（任意）: Rubber Band の CLI（Windows 版の exe が公式配布 [確認: breakfastquay.com]、GPL）で R3 エンジン（`--fine`）を使う。R3 は複雑なミックスや声、低音で R2 より良いが CPU をかなり使う [確認: Rubber Band 文書]。**PC に入れるものなのでユーザー確認が必要**。pyrubberband はこの CLI を呼ぶだけの薄い包み。
- 利点: 再生側は今のエンジンのまま（読み込むファイルが替わるだけ）。iPhone の負担が増えない。T06 のロック画面・オフライン保存とも相性がよい（ただし速度ごとに別ファイルになる）。
- 欠点: 速度を変えてから約 10 秒待つ。速度を連続的に動かす操作には向かない。

#### (c) 推奨

1. **ピッチも変わる**: `playbackRate`（すぐ効く）。
2. **ピッチを保つ**: **サーバー事前変換（ffmpeg rubberband R2 → Opus、キャッシュ）**。変換中は「準備中」を出し、できたら同じ位置（元の曲の時刻 ÷ r）から切り替える。
3. PC で連続的に速度を動かしたくなったら、第二段階で signalsmith-stretch（MIT、ファイル 1 つ同梱）を PC 限定で足す。iPhone は (b) のまま。
- 使わない: HTMLMediaElement の `preservesPitch`（`<audio>` を 8 本並べると要素同士の同期を保証できない [推測]）。

位置の扱い: 変換済み音声の時刻 t' と元の曲の時刻 t は t = t' × r（実測で数 ms のずれ。必要なら定数で補正）。拍・キュー・ループは**元の曲の時刻で保存**し、表示と再生のときに変換する。

---

## D. 分離品質の改善

### D-1. 今のパイプラインで気づいたこと（「その他」にボーカルが混ざる一因）

`separation/pipeline.py` の `run_plan` [確認: コード]:

- vocals ＝ SW の vocals と Kim の vocals の平均（standard）。他の stem は SW のまま。
- 残差 ＝ 元の曲 − Σ（上位 stem）を **other に足す**。
- SW の 6 stem の合計はほぼ元の曲になるので、残差 ≈ SW の vocals − 平均の vocals ＝ (SW vocals − Kim vocals) / 2。
- つまり、**2 つのボーカルモデルの差（主にサブボーカル・リバーブ・息など、片方だけが拾った部分）が「その他」に入る**（片方だけが拾った成分の半分、または逆相の成分）。これが「サブボーカルが他の stem に残る」「その他に多く入りすぎ」の一因と考えられる [推測: 仕組みから。実データで残差を聴いて確認すべき]。
- 対策案: (1) vocals の平均で生じた差を other ではなく **vocals 側（backing）に戻す**（＝「元の曲 − SW の楽器 stem の合計」を vocals とし、Kim は lead/backing の判定の補助に使う）、(2) 残差を stem ごとに記録して聴けるようにし、どこに戻すのが良いか聴き比べる。T09（実機調整）の最初の作業にするとよい。

### D-2. lead / backing（メイン・サブボーカル）

MVSEP の評価（lead / back vocals の SDR、dB）[確認: mvsep.com/algorithms/76]:

| モデル | lead | back | audio-separator 0.47.0 | 状態 |
| --- | --- | --- | --- | --- |
| BS Roformer（MVSep Team） | **10.41** | 6.61 | × | MVSEP のサイト専用 |
| **BS Roformer（anvuew）** | **10.22** | — | ○ `bs_roformer_karaoke_anvuew.ckpt` | 未ダウンロード。ライセンス GPL-3.0 [確認: Hugging Face] |
| BS Roformer（frazer/becruily） | 10.10 | — | ○（standard・best で使用中） | ダウンロード済み |
| MelBand（Fused gabox/aufr33） | 9.85 | — | × | — |
| MelBand（gabox） | 9.67 | — | ○ `mel_band_roformer_karaoke_gabox.ckpt`（v2 もある） | 未ダウンロード |
| MelBand（becruily） | 9.61 | — | ○（fast・best で使用中） | ダウンロード済み |
| MelBand（aufr33/viperx） | 9.45 | — | ○ | — |
| 同 各モデルを「ボーカルを先に抜いてから」使用 | 8.98〜9.62 | 4.98〜5.63 | — | — |

読み取れること:

- MVSEP の表では、**karaoke モデルを元の曲に直接かけた方が lead の SDR が高く**、「ボーカルを先に抜いてから」は lead が 0.2〜0.6 dB 下がる（back は測れるようになる）[確認: 同表]。今の stemapp は「先に抜いてから」方式。
- 案: lead ＝ karaoke を**元の曲に**かけた出力、backing ＝ vocals − lead（今と同じ残差）。両方式を 5 曲程度で聴き比べて決める。
- 案: karaoke を anvuew（10.22）＋ frazer（10.10）の平均にする（どちらも BS-Roformer）。best には gabox を足してもよい。
- メインと同じ旋律を重ねた声（ダブル）やユニゾンのコーラスは、モデルの学習上「lead」に入りやすく、原理的に分けにくい [推測]。「メインに入るべきものがサブに入る」は、karaoke モデルが lead と判断しきれなかった部分が backing（＝vocals − lead）に回るため。どちらにもモデルの判断の限界がある。

audio-separator 0.47.0 の一覧で karaoke 系はほかに `UVR_MDXNET_KARA(_2).onnx`、`5_HP/6_HP-Karaoke-UVR.pth`（古い世代で SDR が低い）[確認: `--list_models`]。

ボーカル全体を広く拾うモデル（backing が他の stem に残る対策）: 一覧に `mel_band_roformer_vocal_fullness_aname.ckpt`、Gabox の FV 系、`mel_band_roformer_vocals_becruily.ckpt`、`melband_roformer_big_beta6x.ckpt` などがある [確認: `--list_models`]。MVSEP の表では becruily deux が vocals SDR 11.35 で、Kim の 11.01 より高く、「fullness」（音を削らない度合い）も大きい [確認: mvsep.com/algorithms/49]。standard の Kim を becruily 系に替えると、サブボーカルを vocals 側に多く拾える可能性がある [推測]。

### D-3. 「その他」をさらに分ける（パッド・オーケストラヒット）

**audio-separator 0.47.0 で使えるもの** [確認: `--list_models`]: 管楽器（`17_HP-Wind_Inst-UVR.pth`、木管）、`kuielab_a/b_other.onnx`（other 全体）程度。**strings・synth・pad・keys 用のモデルは無い**。また audio-separator は一覧に無いモデルファイルを読み込まない（`download_model_files` が「supported model files に無い」とエラーにする）[確認: separator.py]。一覧外のモデルを使うには、ZFTurbo の MSST を直接使うか、audio-separator に手を入れる必要がある。

| 手段 | 分けられるもの | 動く場所・VRAM | ライセンス | 評価 |
| --- | --- | --- | --- | --- |
| MVSep Mega 53 stems（MSST 形式、BS-Roformer） | strings、bowed_strings、synth、keys、organ、brass、wind など 53 種 [確認: MSST v1.0.21] | **VRAM 16GB 以上を推奨**（batch 1 でも）[確認]。8GB では チャンクを縮めても動くか不明 [推測]。各 stem の合計は元に一致しない [確認] | 不明（要確認） | 個別の専用モデルより質は下がると作者が明記 [確認]。T07 の予定どおり、試すなら別枠 |
| MVSEP のサイト（MVSep Synth、Keys、Organ 等） | Synth（Synth Pad、Synth Strings を含む）、Keys、Organ など [確認: mvsep.com] | 外部サービス（アップロードが必要） | サイトの規約 | 重みは公開されていない模様 [推測]。**外部サービス利用はユーザー確認が必要** |
| **HPSS（信号処理）で other を分ける** | 「伸びる音（パッド・持続音）」と「短い音（ヒット・打撃）」＋残り | CPU のみ。librosa 導入済み。184 秒の other で 47 秒 [確認: 実測] | ISC | 楽器の種類ではなく音の「形」で分ける。パッドとオーケストラヒットの分離には合っている [推測: 要試聴] |

- HPSS（Harmonic–Percussive Source Separation、時間方向に長く続く成分と、周波数方向に広がる短い成分を分ける手法）は librosa の `decompose.hpss` にあり、`margin` を大きくすると「どちらでもない残り」を 3 つ目として取れる（Driedger らの拡張）。手元で other に kernel（約 1 秒 × 約 300Hz）、margin=2 をかけると、エネルギーの約 40% が「持続」、11% が「短い音」、13% が「残り」に分かれた [確認: 実測。音質は未試聴]。合計＝元の other になるよう「残り」を必ず作れるので、SPEC の「子の合計＝親」の規則にも合う。
- **オーケストラヒットが原理的に分けにくい理由**: オーケストラヒットは弦・金管・木管（とティンパニ等）の和音を一斉に短く鳴らした音で、もともと 1 つのサンプルとして作られ使われてきた [確認: Wikipedia "Orchestra hit"]。楽器の種類で分けるモデルにとっては「弦」「金管」「シンセ」のどれでもあり、どれでもない。学習データの stem の区分にも普通は無い。一方で、パッドは「長く伸びる」、ヒットは「短い」という時間の形がはっきり違うので、**楽器の種類ではなく時間の形で分ける（HPSS 等）方が見込みがある** [推測]。
- 推奨: (1) T07 の詳細分割に「other → 持続音（パッド）/ 短い音（ヒット等）/ 残り」を HPSS で加える（GPU 不要、追加インストール不要）。(2) それでも足りなければ Mega 53 を別ブランチで 8GB で動くか試す。(3) MVSEP の利用は外部サービスなのでユーザーの判断。

---

## E. まとめ

### E-1. 推奨構成

- **拍検出**: beat_this（MIT）を分割後に 1 回。元の曲を入力。結果は `BEAT_GRID` に保存（自動結果とユーザーのアンカーを分ける）。予備は librosa。
- **表示**: 拍の線（細）・小節線（太・番号）を 1 秒目盛りの代わりに描く。区間ごとの BPM を表示。
- **速度変更**: 「ピッチも変わる」＝ playbackRate。「ピッチを保つ」＝ サーバーで ffmpeg rubberband → Opus をキャッシュ。拍・キューは元の曲の時刻で持つ。
- **品質**: (1) 残差を other に足すやり方の見直し、(2) karaoke を anvuew＋frazer にし「元の曲に直接」方式と比べる、(3) other を HPSS で持続音／短い音に分ける。

### E-2. 実装タスクへの分け方（提案）

| ID（仮） | 内容 | 依存 | 必要な追加 |
| --- | --- | --- | --- |
| T10a | 拍の解析: beat_this を後処理に追加、`BEAT_GRID` 表、API。Fake の拍解析器でテスト | — | beat-this、torchaudio 2.11、重み 78MB |
| T10b | 拍・小節線の描画、区間 BPM の表示 | T10a | — |
| T10c | 補正 UI: 1 小節目をここに、×2/÷2、拍子、タップ、キュー 2 点から計算（アンカー） | T10b | — |
| T11a | ピッチも変わる速度変更（playbackRate、位置計算、ループ・シークとの整合） | — | — |
| T11b | ピッチを保つ速度変更（サーバー変換ジョブ、キャッシュ、切り替え） | T11a | —（ffmpeg にある） |
| T12 | 分離の見直し: 残差の行き先、karaoke の入力と組み合わせ、聴き比べ用の CLI | — | karaoke anvuew / gabox の重み（各数百 MB〜1.7GB） |
| T07 に追加 | other → 持続音／短い音／残り（HPSS） | — | — |

T11a は小さく、すぐ効果が見えるので最初でもよい。

### E-3. ユーザーに確認すること

1. **beat_this の導入**: Python パッケージ `beat-this` と `torchaudio 2.11`（数 MB〜）、重み 78MB のダウンロード。すべて MIT（torchaudio は BSD）。
2. **ピッチを保つ速度変更の方式**: 「速度を変えて約 10 秒待つ（高品質・iPhone に優しい）」でよいか。PC だけでも「すぐ変わる」方が欲しいか（その場合 signalsmith-stretch のファイル 1 つを同梱、MIT）。
3. **より高品質な伸縮（任意）**: Rubber Band の CLI（Windows 版の exe、GPL）を入れて R3 エンジンを使うか。R2 との差を先に聴き比べることを勧める。
4. **karaoke モデルの追加ダウンロード**: anvuew（GPL-3.0）、gabox。容量は各数百 MB〜。
5. **「その他」の細分化**: まず HPSS（追加なし）で試してよいか。Mega 53（VRAM 16GB 推奨、8GB で動くか不明）や MVSEP（外部サービスに曲をアップロードする）を使うか。
6. 速度変換のキャッシュ容量の上限（例: 1 曲あたり速度 3 つまで、全体 2GB）。

---

## 出典

- beat_this: https://github.com/CPJKU/beat_this 、https://pypi.org/project/beat-this/ 、論文 https://arxiv.org/abs/2407.21658（表2 は https://arxiv.org/html/2407.21658v1）、依存 https://raw.githubusercontent.com/CPJKU/beat_this/main/pyproject.toml 、https://raw.githubusercontent.com/CPJKU/beat_this/main/beat_this/preprocessing.py
- torchaudio 2.11 が最後の版: https://docs.pytorch.org/audio/stable/index.html 、https://github.com/pytorch/audio/releases
- All-In-One: https://github.com/mir-aidj/all-in-one 、https://pypi.org/project/allin1/ 、論文 https://arxiv.org/abs/2307.16425
- NATTEN の導入: https://natten.org/install/
- madmom: https://github.com/CPJKU/madmom 、Py3.10 以上の問題 https://github.com/CPJKU/madmom/issues/523 、https://github.com/xavriley/crepe_notes/issues/15
- BeatNet: https://github.com/mjhydri/BeatNet
- librosa（可変テンポ）: https://librosa.org/doc/main/auto_examples/plot_dynamic_beat.html 、https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html
- Beat Transformer（分離入力）: https://arxiv.org/abs/2209.07140
- 分離後 drums の onset: https://arxiv.org/abs/2609.04224
- rekordbox: https://www.lexicondj.com/blog/understanding-rekordbox-beatgrid-analysis 、https://community.pioneerdj.com/hc/en-us/community/posts/22976973667737-Rekordbox-Dynamic-Beat-Grid-Editing-Enhancements
- Serato: https://support.serato.com/hc/en-us/articles/360001274936-Beatgrids 、https://support.serato.com/hc/en-us/articles/227627028-Slip-Incorrect-Beatgrid 、https://the-drop.serato.com/announcements/introducing-lucid-beatgrids-in-serato-dj-5-0/
- Traktor Pro 4: https://www.digitaldjtips.com/reviews/traktor-pro-4-review/
- Ableton: https://www.ableton.com/en/manual/audio-clips-tempo-and-warping/
- playbackRate: https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode/playbackRate 、preservesPitch: https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/preservesPitch
- signalsmith-stretch: https://github.com/Signalsmith-Audio/signalsmith-stretch 、https://raw.githubusercontent.com/Signalsmith-Audio/signalsmith-stretch/main/web/release/README.md 、https://raw.githubusercontent.com/Signalsmith-Audio/signalsmith-stretch/main/web/web-wrapper.js
- SoundTouchJS: https://github.com/cutterbl/SoundTouchJS
- AudioWorklet の対応状況: https://caniuse.com/mdn-api_audioworklet 、iOS の音割れ報告 https://github.com/WebAudio/web-audio-api/issues/2632
- Rubber Band: https://breakfastquay.com/rubberband/ 、R3 エンジン https://breakfastquay.com/rubberband/code-doc/classRubberBand_1_1RubberBandStretcher.html 、ffmpeg フィルタ https://ffmpeg.org/ffmpeg-filters.html#rubberband
- MVSEP: 一覧 https://mvsep.com/en/algorithms 、Karaoke https://mvsep.com/algorithms/76 、BS Roformer SW https://mvsep.com/algorithms/77 、MelBand Roformer https://mvsep.com/algorithms/49 、Synth https://mvsep.com/algorithms/85
- anvuew karaoke（ライセンス）: https://huggingface.co/anvuew/karaoke_bs_roformer
- MSST / Mega 53: https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/tag/v1.0.21 、https://huggingface.co/noblebarkrr/BS-Roformer-MVSep-Mega-53-stems
- オーケストラヒット: https://en.wikipedia.org/wiki/Orchestra_hit
- HPSS: https://librosa.org/doc/latest/generated/librosa.decompose.hpss.html （Driedger, Müller, Disch 2014 "Extending harmonic-percussive separation of audio signals"）

## 付録: 手元で確かめたこと

- `.venv`: Python 3.12.14、torch 2.11.0+cu128、librosa 1.0.0、numba 0.67.0、scipy 1.18.1、soxr 1.1.0、einops 0.8.2、rotary-embedding-torch 0.6.5、tqdm 4.70.1、resampy 0.4.3。**torchaudio・madmom・beat_this は無い。**
- ffmpeg 8.0 full build: `rubberband`、`atempo`、`asetrate`、`aresample` あり。
- `audio-separator --list_models`（0.47.0）で karaoke・vocals・その他のモデルを確認（ダウンロードはしていない）。ダウンロード済みは SW、Kim vocals、karaoke becruily（Mel）、karaoke frazer（BS）の 4 つ。
- 計測に使った一時ファイルは scratchpad に置いた（data 以下は読んだだけ）。
