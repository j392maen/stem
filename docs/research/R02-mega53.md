# R02 MVSep Mega 53 stems を RTX 4060 Laptop（8GB）で動かす

T07b の調査（指示書の 1）と試用（2）の記録。調査日 2026-10-02。
[確認] は一次情報（配布元のファイル・API・コード）で確かめたこと、[推測] は確かめていないこと。

## 1. 調査

### 1-1. 重み・設定ファイル

| 配布元 | ファイル | サイズ | 備考 |
| --- | --- | --- | --- |
| ZFTurbo/Music-Source-Separation-Training の GitHub リリース v1.0.21 | `mvsep_mega_model_bs_roformer_53_stems_v1.ckpt` | **1,368,919,887 バイト（約 1.37GB）** | 53 stem すべての重み。[確認: GitHub API の release assets] |
| 同上 | `mvsep_mega_model_bs_roformer_53_stems.yaml` | 4,184 バイト | `model.num_stems: 53`、`training.instruments` に 53 個の名前 [確認] |
| Hugging Face `noblebarkrr/BS-Roformer-MVSep-Mega-53-stems`（第三者の作り直し） | `v1/bs_mega_53stem_<stem>_mvsep.ckpt` ×53 | 各約 77.6MB | 1 stem ずつに切り出したもの（`num_stems: 1`、`instruments: [<stem>, other]`）[確認: HF API・config] |

- **2GB 以下**（1.37GB）。本体（共通の Transformer）＋ stem ごとの「マスク推定器（mask estimator、各 stem の取り出し方を決める部分）」53 個の構成。
  パラメータを数えると本体 約 2,600 万、マスク推定器 1 つ約 1,230 万 × 53 ≒ 6.8 億で、1.37GB は fp16（16bit 浮動小数点）で保存されているとちょうど合う [推測: 計算。試用で確かめる]。
- 正式な配布元（ZFTurbo のリリース）を使う。HF の切り出し版は第三者のもので、stem ごとに本体を重複して持つので、多く使うと合計が大きくなる。
- **重みのライセンス: 明示なし（不明）**。リリースページ・yaml・HF の README のどこにも書かれていない [確認]。
  コードのリポジトリは MIT。MVSep 側の個別モデルはサイト上でのみ提供され重みは非公開 [R01]。
  無償で一般公開されたものを自分の PC で使うだけ（再配布しない）なので、個人利用で問題になる記述は見当たらない [推測]。重みはリポジトリにコミットしない。

### 1-2. 推論に必要なコードとライセンス

- 必要なのは MSST の `models/bs_roformer/bs_roformer.py`（BSRoformer クラス）と `models/bs_roformer/attend.py`（注意機構）の 2 ファイルだけ。
  ライセンスは **MIT**（Copyright (c) 2024 Roman Solovyev (ZFTurbo)）[確認: LICENSE]。
- v1.0.21 の `BSRoformer.forward` は `active_stem_ids`（使う stem の番号の一覧）を受け取れ、**指定した stem のマスク推定器だけを動かせる** [確認: コード]。8GB で動かす鍵になる（1-5）。
- import しているもの: torch、einops、beartype、rotary_embedding_torch、packaging（＋任意の PoPE_pytorch。無ければ使わない）[確認]。
- 今の `.venv` にある: einops 0.8.2、beartype 0.18.5、rotary-embedding-torch 0.6.5、packaging 26.3、torch 2.11.0+cu128、PyYAML 6.0.3 [確認: importlib.metadata]。
  **新しい Python パッケージは不要**。
- 参考: audio-separator 0.47.0 にも BSRoformer の写しがあるが、一覧に無いモデルは読み込まない（R01 D-3）ので使わない。
- チャンク分割と重ね合わせ（overlap-add）は MSST の `utils/model_utils.py` の `demix` に相当する処理が要る。窓・重なりの考え方だけ借りて、stemapp 側で短く書く。

### 1-3. 53 stem の名前と stemapp の STEM_TYPE の対応

yaml の `training.instruments` の順（= 出力の順）[確認]:

accordion, acoustic-guitar, back-vocal, banjo, bass, bassoon, bells, bowed_strings, brass, cello, clarinet, congas, digital-piano, dobro, double-bass, drums, electric-guitar, flute, french-horn, glockenspiel, guitar, harmonica, harp, harpsichord, hh, keys, kick, lead-vocal, mandolin, marimba, oboe, organ, percussion, piano, saxophone, sitar, snare, strings, synth, tambourine, timpani, toms, triangle, trombone, trumpet, tuba, ukulele, viola, violin, vocal, wind, wind-chimes, woodwind

other の子として意味があるもの（親の other は SW 分割の残りで、vocals・drums・bass・guitar・piano は既に抜かれている）:

| Mega 53 の名前 | stemapp の STEM_TYPE（既存） | 備考 |
| --- | --- | --- |
| strings | strings | 弦の合奏。bowed_strings / violin / viola / cello と重なる |
| brass | brass | trumpet / trombone / tuba / french-horn と重なる |
| woodwind | woodwind | flute / clarinet / oboe / bassoon / saxophone と重なる [推測] |
| saxophone | saxophone | woodwind と重なる [推測] |
| wind | wind | brass ＋ woodwind をまとめたもの [推測: 名前から] |
| synth | synth | |
| keys | keys | piano / organ / digital-piano と重なる [推測] |
| organ | organ | |
| percussion | percussion | |
| bells, glockenspiel, congas, tambourine, triangle | percussion の子（既存） | |
| その他（harp, accordion, timpani, marimba, ...） | 無し | 必要なら追加 |

- 作者の注記: 「各 stem の合計は元に一致しない。重なる情報を持つものがある（例: vocals は lead と back の両方を含む）」[確認: リリースノート]。
  したがって、子に使う stem は**互いに重ならない組**を選ぶ必要がある（重なる 2 つを両方子にすると同じ音が 2 回入り、「残り」がそれを打ち消す逆相の音になる）。組は試用の結果で決める（2 章）。

### 1-4. 公表されている必要 VRAM

- 「batch 1 でもメモリを多く使う。16GB 以上を勧める」[確認: リリースノート]。
- 推論設定: `inference.chunk_size: 882000`（44.1kHz で 20 秒）、`num_overlap: 2`、`batch_size: 1`、学習は `use_amp: true`（fp16 混合精度）[確認: yaml]。

### 1-5. 8GB で動かす工夫（見込み）

VRAM の大部分は 53 個のマスク推定器の中間値と出力（stem ごとに 2ch × 1025 周波数 × 約 1,700 フレームの複素マスク）と、fp32 に広げた重み（約 2.7GB）[推測: 計算]。次の順で減らす。

1. **使う stem のマスク推定器だけ動かす**（`active_stem_ids`）。other の子に使う 10 個前後なら、出力と中間値は 53 個の約 1/5。使わない推定器は GPU に載せない（重みも約 1/5）。
2. **fp16 の autocast**（学習も fp16 混合精度なので質の低下は小さい [推測]）。
3. チャンクを短くする（20 秒 → 10 秒 → 5 秒）。足りないときだけ。
4. batch 1。

結論: 新しいパッケージは不要、重みは 1.37GB（2GB 以下）、コードは MIT、重みのライセンスは明示なし（個人利用で問題となる記述なし）。指示書の条件を満たすので 2（試用）に進む。

## 2. 試用（RTX 4060 Laptop 8GB、2026-10-02）

ユーザーの曲の other（SW 分割の残り）を一時フォルダにコピーして実行した（実データは読み取りのみ）。
重みは `data/models/mvsep_mega_model_bs_roformer_53_stems_v1.ckpt`（1.37GB、中身は fp16 で 13,595 個のテンソル、
パラメータ 6.82 億のうち 6.55 億がマスク推定器 [確認]）。推論は fp16 の autocast、batch 1、窓の重なり 2。

### 2-1. 時間と GPU メモリ（184 秒の曲「水槽」の other）

| 動かした stem | チャンク | 読み込み | 推論 | GPU 最大（allocated / reserved） |
| --- | --- | --- | --- | --- |
| 53 個すべて | 20 秒（既定） | 8.1 秒 | 66.3 秒 | **9,155 / 10,504 MB**（8GB を超える。Windows が共有メモリにあふれさせて完走した [推測]） |
| 53 個すべて | 10 秒（0.5 倍） | 6.7 秒 | 61.8 秒 | 5,886 / 6,578 MB |
| 53 個すべて | 5 秒（0.25 倍） | 6.7 秒 | 85.7 秒 | 4,251 / 4,664 MB |
| 2 個（strings, synth） | 20 秒 | 6.3 秒 | 30.9 秒 | 1,446 / 1,646 MB |
| **5 個（採用した組）** | 20 秒 | 6.8 秒 | 32.7 秒 | **1,587 / 1,796 MB** |

- OOM（メモリ不足のエラー）は一度も出なかった。53 個すべてでも 0.5 倍のチャンクなら 8GB に収まる。
- **使う stem のマスク推定器だけを動かす方式なら 1.6GB** で、チャンクを縮める必要はない。stemapp の詳細分割はこの方式（5 個）にした。
- stemapp の詳細分割（`run_refine`、本物の分離器）で通して: 184 秒の曲 42.9 秒、154 秒の曲「春嵐」36.6 秒、どちらも GPU 最大 1,587MB。子の合計＝親（差 0）。

### 2-2. 各出力の RMS（dBFS。53 個すべて、チャンク 0.5 倍）

「水槽」（other の RMS −28.7dB）で −60dB 以上のもの（多い順）:
synth −34.3、keys −38.5、marimba −42.9、organ −43.5、vocal −44.3、strings −45.7、back-vocal −46.5、percussion −46.9、accordion −48.1、violin −48.2、bowed_strings −49.5、bass −49.6、cello −52.7、double-bass −53.0、bells −53.9、dobro −56.4、lead-vocal −56.5、acoustic-guitar −56.6、sitar −59.2、wind −59.6、brass −59.6、banjo −59.8。
ほかの 31 個は −60dB 未満（drums −100.4、snare −98.8、bassoon −95.7 など）。

「春嵐」（other −25.0dB）で −60dB 以上のもの: synth −26.3、keys −34.2、violin −35.7、bowed_strings −36.7、bass −36.7、strings −39.9、accordion −49.6、vocal −49.8、double-bass −50.6、piano −51.2、organ −51.3、back-vocal −51.8、digital-piano −52.9、harpsichord −53.6、flute −58.0。

stemapp の詳細分割（採用した 5 個、20 秒チャンク）での子の RMS:

| 曲 | brass | woodwind | strings | synth | percussion | 残り |
| --- | --- | --- | --- | --- | --- | --- |
| 水槽（親 −28.7） | −57.7 | 無音（作らない） | −53.0 | −33.9 | −43.8 | −32.9 |
| 春嵐（親 −25.0） | −53.7 | 無音（作らない） | −39.1 | −26.4 | 無音（作らない） | −36.8 |

### 2-3. 重なりと、子にする組の決め方

出力どうしの正規化内積（1 なら同じ音、0 なら無関係）が大きい組 [確認: 実測、水槽 / 春嵐]:
keys–organ 0.83 / 0.59、keys–synth 0.72 / 0.63、organ–synth 0.50 / 0.34、brass–wind 0.90、back-vocal–vocal 0.93 / 0.95、bowed_strings–violin（春嵐）0.98、strings–bowed_strings 0.73 / 0.93、marimba–percussion 0.82。
→ keys・wind・vocal は「まとめ」の stem で、細かい stem と同じ音を持つ。

候補の組を other に当てはめた結果（「重なり」= 子の合計のエネルギー − 子のエネルギーの合計。同じ音が 2 回入った分。親に対する %）:

| 組 | 水槽: 残り | 水槽: 重なり | 春嵐: 残り | 春嵐: 重なり |
| --- | --- | --- | --- | --- |
| strings, brass, woodwind, synth, keys, organ, percussion | 47.7% | +46.5% | 15.9% | +46.2% |
| strings, brass, woodwind, synth, organ, percussion | 37.2% | +12.0% | 5.9% | +6.5% |
| **strings, brass, woodwind, synth, percussion（採用）** | 37.9% | **+2.4%** | 5.9% | **+3.6%** |
| bowed_strings, brass, woodwind, synth, percussion | 39.0% | +1.5% | 5.4% | +7.7% |

- keys・organ を入れると同じ音が 2 回入り、「残り」がそれを打ち消す逆相の音になる。重なりの少ない **strings / brass / woodwind / synth / percussion** を子にした（`stemapp.seed.MEGA53_CHILDREN`）。organ・keys・marimba・ボーカルの漏れなどは「残り（その他）」に入る。
- 「水槽」では残りが親のエネルギーの 38%（marimba・organ・ボーカルの漏れ・accordion など）。「春嵐」では 6%。
- **チャンクの長さで結果が変わる**: 「水槽」の strings は 20 秒チャンクで −51.9dB、10 秒で −45.7dB、organ は −46.6 → −43.5dB。学習は 10 秒（`audio.chunk_size: 441000`）、推論の既定は 20 秒（`inference.chunk_size: 882000`）。どちらが良いかは聴かないと分からないので、作者の推論の既定（20 秒）のままにした [要試聴]。

### 2-4. 判断

8GB で問題なく動き（1.6GB・3 分の曲で約 40 秒）、synth（パッド等）が other の大部分を取り出せている（水槽 27%、春嵐 75% のエネルギー）。実用になると判断して、T07 の詳細分割の方法に組み込んだ（3）。分かれ方の質（オーケストラヒットがどこに入るか等）は試聴で確かめる必要がある。
