"""ZFTurbo の Music-Source-Separation-Training（MSST）から取り込んだ推論コード（T07b）。

取得元: https://github.com/ZFTurbo/Music-Source-Separation-Training タグ v1.0.21
- `bs_roformer.py` ← models/bs_roformer/bs_roformer.py
- `attend.py` ← models/bs_roformer/attend.py
ライセンス: MIT（同じフォルダの LICENSE。Copyright (c) 2024 Roman Solovyev (ZFTurbo)）。

元からの変更点（`# stemapp:` の印を付けた行）:
- import 先（`models.bs_roformer.attend` → `.attend`）
- attend.py: 非推奨の `torch.backends.cuda.sdp_kernel` を `torch.nn.attention.sdpa_kernel` に、
  標準出力への print をログに置き換えた

torch・einops・beartype・rotary_embedding_torch を import するので、GPU 依存の無い環境では
このパッケージを import しないこと（`runner` は関数の中で import する）。
"""
