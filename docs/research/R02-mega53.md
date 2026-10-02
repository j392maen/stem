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
