"""python-audio-separator を使う分離器。

`audio_separator` と `torch` を import するのはこのモジュールの中だけ（GPU 依存の無い環境でも
他のコードとテストが動くように、import は関数の中で行う）。

audio-separator の `Separator.separate()` は入力と各出力を個別に正規化（ピーク 0.9 に縮める）
してファイルに書くため、stem の合計が元の曲と一致しなくなる。そこでモデルのロードだけ
audio-separator に任せ、推論はロード済みインスタンスの `demix()` を直接呼んで、
正規化せずに配列のまま受け取る。

audio-separator の一覧に無い MSST 形式のモデル（MVSep Mega 53 stems）は、取り込んだ MSST の
推論コード（`stemapp.separation.msst`）で動かす（T07b）。
"""

from __future__ import annotations

import gc
import hashlib
import io
import logging
import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stemapp.audio import read_audio
from stemapp.seed import ASPIRATION, DRUMSEP, MALE_FEMALE, MEGA53
from stemapp.separation.base import (
    DEVICE_CPU,
    DEVICE_CUDA,
    OPT_CHUNK_SCALE,
    OPT_OVERLAP,
    OPT_SEGMENT_SIZE,
    OPT_TTA,
)

log = logging.getLogger(__name__)

# audio-separator の出力名（ファイル名の "(Vocals)" の部分。小文字で比較）→ stem 名。
# role ごとに持つ。モデルの yaml（training.instruments）で確認できた名前だけを載せる。
# - multistem: BS-Roformer-SW.yaml → bass, drums, other, vocals, guitar, piano
# - vocals:    vocals_mel_band_roformer.yaml → vocals, other
#              （mel_band_roformer_kim_ft_unwa も --list_models の stems は vocals, other）
# - karaoke:   config_mel_band_roformer_karaoke_becruily.yaml /
#              config_bs_roformer_karaoke_frazer_becruily.yaml → Vocals, Instrumental
#              karaoke は "(Vocals)"=lead、"(Instrumental)"=それ以外。
# 表に無い名前が出たらエラーにする（新しいモデルを足すときに yaml を見て追記する）。
OUTPUT_NAME_MAP: dict[str, dict[str, str]] = {
    "multistem": {
        "vocals": "vocals",
        "drums": "drums",
        "bass": "bass",
        "guitar": "guitar",
        "piano": "piano",
        "other": "other",
    },
    "vocals": {
        "vocals": "vocals",
        "other": "instrumental",
    },
    "karaoke": {
        "vocals": "lead_vocal",
        "instrumental": "backing_vocal",
    },
}

# 詳細分割（role=refine）はモデルごとに出力名が違うので、モデルのファイル名で引く。
# yaml（training.instruments）で確かめた名前だけを載せる（T07）:
# - MDX23C-DrumSep-aufr33-jarredou.ckpt（config_drumsep_mdx23c.yaml）
#   → kick, snare, toms, hh, ride, crash
# - bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt（config_chorus_male_female_bs_roformer.yaml）
#   → male, female
# - aspiration_mel_band_roformer_sdr_18.9845.ckpt（config_aspiration_mel_band_roformer.yaml）
#   → aspiration, other（other は「息以外」。子には使わず、残りは親 − 息 で作る）
ROLE_REFINE = "refine"
REFINE_OUTPUT_NAME_MAP: dict[str, dict[str, str]] = {
    DRUMSEP: {
        "kick": "kick",
        "snare": "snare",
        "toms": "toms",
        "hh": "hihat",
        "ride": "ride",
        "crash": "crash",
    },
    MALE_FEMALE: {
        "male": "male",
        "female": "female",
    },
    ASPIRATION: {
        "aspiration": "breath",
        "other": "no_breath",
    },
    # mvsep_mega_model_bs_roformer_53_stems.yaml（training.instruments の 53 個のうち、
    # other の子に使うものだけ。ここに載せた名前のマスク推定器だけを動かす。R02）
    MEGA53: {
        "strings": "strings",
        "brass": "brass",
        "woodwind": "woodwind",
        "synth": "synth",
        "percussion": "percussion",
    },
}


@dataclass(frozen=True)
class MsstModelFiles:
    """audio-separator を通さずに動かす MSST 形式のモデル（重み・設定の入手先と検証値）。

    size・sha256 はダウンロードしたファイルの検証に使う（既にあるファイルはサイズだけ見る）。
    """

    config: str
    url: str
    config_url: str
    size: int
    sha256: str
    config_size: int
    config_sha256: str
    size_label: str  # 画面に出す大きさ（例 "約1.4GB"）


_MSST_RELEASE = (
    "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.21"
)
MSST_MODELS: dict[str, MsstModelFiles] = {
    # サイズ・SHA-256 は 2026-10-02 に v1.0.21 のリリースから入手したファイルで求めた値（R02）
    MEGA53: MsstModelFiles(
        config="mvsep_mega_model_bs_roformer_53_stems.yaml",
        url=f"{_MSST_RELEASE}/{MEGA53}",
        config_url=f"{_MSST_RELEASE}/mvsep_mega_model_bs_roformer_53_stems.yaml",
        size=1_368_919_887,
        sha256="c62820893bbf86d4e734f966bd142d9157cfc8bb8e79e9d8f9ea553f3ff3519f",
        config_size=4_184,
        config_sha256="7e198062a251587088adb91215a4f44ab59e67bd62fcc805cf54d6e7dfc51103",
        size_label="約1.4GB",
    ),
}

# ダウンロード: (url, 書き込み先, 進み具合（受け取ったバイト数, 全体のバイト数 or None）)
ByteProgress = Callable[[int, int | None], None]
Downloader = Callable[[str, Path, ByteProgress], None]
StageProgress = Callable[[float, str], None]


def http_download(url: str, dest: Path, on_bytes: ByteProgress) -> None:
    """url を dest に書く（1MB ずつ。Content-Length があれば全体の大きさを渡す）"""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "stemapp"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:  # noqa: S310
        length = resp.headers.get("Content-Length")
        total = int(length) if length and length.isdigit() else None
        done = 0
        on_bytes(done, total)
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            on_bytes(done, total)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

MIN_SEGMENT = 32  # チャンクを縮めるときの下限（dim_t）


def map_outputs(
    role: str, outputs: Mapping[str, np.ndarray], model_filename: str | None = None
) -> dict[str, np.ndarray]:
    """audio-separator の出力名を stem 名に置き換える。表に無い名前はエラー。

    role=refine のときは model_filename で表を選ぶ。
    """
    if role == ROLE_REFINE:
        table = REFINE_OUTPUT_NAME_MAP.get(model_filename or "")
        if table is None:
            raise ValueError(f"詳細分割の出力名の表に無いモデルです: {model_filename}")
    else:
        table = OUTPUT_NAME_MAP.get(role)
    if table is None:
        raise ValueError(f"未知の role: {role}")
    mapped: dict[str, np.ndarray] = {}
    for name, arr in outputs.items():
        key = table.get(name.strip().lower())
        if key is None:
            raise RuntimeError(
                f"モデルの出力「{name}」を stem に対応づけられません（role={role}）。"
                "OUTPUT_NAME_MAP に追加してください。"
            )
        if key in mapped:
            raise RuntimeError(f"出力「{name}」が {key} に重複して対応づけられました。")
        mapped[key] = arr
    return mapped


# onnxruntime の preload_dlls() が標準出力に print する、CUDA の版の違いの注意と
# DLL の読み込み失敗。
# onnxruntime-gpu（CUDA 13 用）と torch（cu128）の版が違うために出る。stemapp のモデルはすべて
# torch（.ckpt）で動き、ONNX のモデルは使わないので実害は無い（T14 で確かめた）。
_ORT_NOISE = (
    "uses CUDA",
    "Failed to load ",
    "Please follow https://onnxruntime.ai",
    "Skip loading CUDA and cuDNN DLLs",
)
ORT_NOTE = (
    "onnxruntime-gpu（CUDA 13 用）と torch（CUDA 12.8 用）の版が違うため、onnxruntime の "
    "CUDA の DLL は読み込めません。stemapp のモデルはすべて torch で動くので影響はありません。"
)
_ort_noted = False


def _is_ort_noise(line: str) -> bool:
    return any(key in line for key in _ORT_NOISE)


@contextmanager
def quiet_onnxruntime_preload() -> Iterator[None]:
    """audio-separator の準備（onnxruntime の preload_dlls）が出す注意を1行にまとめる。

    標準出力への print を受け取り、onnxruntime の版の違いによる注意は最初の1回だけ
    ORT_NOTE をログに出す（2回目からは出さない）。それ以外の出力はそのままログに出す。
    """
    global _ort_noted
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            yield
    finally:
        noisy = False
        for raw in buf.getvalue().splitlines():
            line = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()
            if not line:
                continue
            if _is_ort_noise(line):
                noisy = True
            else:
                log.info("%s", line)
        if noisy and not _ort_noted:
            _ort_noted = True
            log.info(ORT_NOTE)


Demix = Callable[[np.ndarray], dict[str, np.ndarray]]


def tta_combine(demix: Demix, mix: np.ndarray) -> dict[str, np.ndarray]:
    """TTA: 元の音・位相反転・左右入れ替えの3通りで分け、元に戻して平均する。

    demix は (samples, 2) を受け取り {名前: (samples, 2)} を返す関数。
    """
    base = demix(mix)
    inv = demix(-mix)
    swp = demix(np.ascontiguousarray(mix[:, ::-1]))
    return {
        k: ((v - inv[k] + swp[k][:, ::-1]) / 3.0).astype(np.float32) for k, v in base.items()
    }


class AudioSeparatorBackend:
    """audio-separator でモデルを1つずつロードし、使い終わったら解放する分離器。"""

    def __init__(
        self,
        models_dir: Path,
        work_dir: Path,
        use_fp16: bool = True,
        downloader: Downloader = http_download,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.work_dir = Path(work_dir)
        self.use_fp16 = use_fp16
        self.downloader = downloader  # テストでは差し替える（ネットワークを使わない）

    # --- GPU メモリ計測 -------------------------------------------------------------

    def reset_peak_memory(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def peak_memory_mb(self) -> float | None:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    # --- 分離 ----------------------------------------------------------------------

    def _load(self, model_filename: str, device: str) -> tuple[Any, Any]:
        """audio-separator の Separator を作り、モデルをロードして (separator, instance) を返す。"""
        import torch
        from audio_separator.separator import Separator as AsSeparator

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if device == DEVICE_CUDA and not torch.cuda.is_available():
            raise RuntimeError("CUDA が使えません（--cpu で CPU 実行できます）。")
        with quiet_onnxruntime_preload():
            sep = AsSeparator(
                log_level=logging.WARNING,
                model_file_dir=str(self.models_dir),
                output_dir=str(self.work_dir),
                # 推論は demix() を直接呼び、autocast は自分でかける
                use_autocast=False,
            )
        if device == DEVICE_CPU:
            sep.torch_device = sep.torch_device_cpu
        try:
            sep.load_model(model_filename)  # 無ければ models_dir に自動でダウンロードされる
        except SystemExit as e:  # audio-separator はロード失敗時に sys.exit(1) する
            raise RuntimeError(
                f"モデル {model_filename} のロードに失敗しました（ファイル破損の可能性）。"
            ) from e
        instance = sep.model_instance
        if instance is None or not hasattr(instance, "demix"):
            raise RuntimeError(f"モデル {model_filename} は demix に対応していません。")
        return sep, instance

    @staticmethod
    def _release(sep: Any, instance: Any) -> None:
        import torch

        try:
            if sep is not None:
                sep.model_instance = None
            if instance is not None and hasattr(instance, "model_run"):
                instance.model_run = None
        finally:
            del sep, instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def separate(
        self,
        wav_path: Path,
        model_filename: str,
        options: Mapping[str, Any],
        device: str,
        role: str,
    ) -> dict[str, np.ndarray]:
        import torch

        if model_filename in MSST_MODELS:
            return self._separate_msst(wav_path, model_filename, options, device, role)
        mix = read_audio(wav_path)  # (samples, 2)
        n = mix.shape[0]
        sep = instance = None
        try:
            sep, instance = self._load(model_filename, device)
            cfg = instance.model_data_cfgdict
            base_segment = int(options.get(OPT_SEGMENT_SIZE) or cfg.inference.dim_t)
            scale = float(options.get(OPT_CHUNK_SCALE, 1.0))
            segment = max(MIN_SEGMENT, int(base_segment * scale))
            instance.segment_size = segment
            if options.get(OPT_OVERLAP):
                instance.overlap = int(options[OPT_OVERLAP])
            log.info(
                "%s: device=%s segment=%d overlap=%s tta=%s",
                model_filename, device, segment, instance.overlap, bool(options.get(OPT_TTA)),
            )

            use_autocast = self.use_fp16 and device == DEVICE_CUDA
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_autocast
                else nullcontext()
            )

            def demix(x: np.ndarray) -> dict[str, np.ndarray]:
                # demix は (2, samples) を受け取り {名前: (2, samples)} を返す
                with ctx:
                    src = instance.demix(
                        mix=np.ascontiguousarray(x.T), override_model_segment_size=True
                    )
                if not isinstance(src, dict):
                    target = cfg.training.target_instrument or "output"
                    src = {target: src}
                return {k: np.asarray(v, dtype=np.float32).T for k, v in src.items()}

            outputs = tta_combine(demix, mix) if options.get(OPT_TTA) else demix(mix)
            mapped = map_outputs(role, outputs, model_filename)
            return {k: _fit(v, n) for k, v in mapped.items()}
        finally:
            self._release(sep, instance)


    # --- MSST 形式（audio-separator の一覧に無いもの） -----------------------------------

    def prepare_model(self, model_filename: str, progress: StageProgress | None = None) -> None:
        """分離の前に要るファイルを用意する（MSST 形式の重みのダウンロード）。

        詳細分割（run_refine）が分離の前に呼び、ダウンロード中の stage・進み具合を受け取る。
        audio-separator のモデルは load_model が自分でダウンロードするので何もしない。
        """
        if model_filename in MSST_MODELS:
            self.ensure_msst_files(model_filename, progress)

    def ensure_msst_files(
        self, model_filename: str, progress: StageProgress | None = None
    ) -> tuple[Path, Path]:
        """重みと設定が無ければダウンロードして検証する（途中は .part に書き、検証後に名前を変える）

        既にあるファイルはサイズだけ確かめ、違えば消してダウンロードし直す。
        ダウンロードしたファイルはサイズと SHA-256 を確かめ、合わなければ消してエラーにする。
        """
        files = MSST_MODELS[model_filename]
        self.models_dir.mkdir(parents=True, exist_ok=True)
        stage = f"モデルをダウンロード中（{files.size_label}）"
        out: list[Path] = []
        for name, url, size, sha in (
            (model_filename, files.url, files.size, files.sha256),
            (files.config, files.config_url, files.config_size, files.config_sha256),
        ):
            path = self.models_dir / name
            if path.is_file():
                actual = path.stat().st_size
                if actual == size:
                    out.append(path)
                    continue
                log.warning(
                    "モデル %s のサイズが違います（%d バイト、正しくは %d）。"
                    "ダウンロードし直します。",
                    path, actual, size,
                )
                path.unlink()
            self._download_verified(url, path, size, sha, stage, progress)
            out.append(path)
        return out[0], out[1]

    def _download_verified(
        self,
        url: str,
        path: Path,
        size: int,
        sha256: str,
        stage: str,
        progress: StageProgress | None,
    ) -> None:
        tmp = path.with_name(path.name + ".part")
        tmp.unlink(missing_ok=True)
        last = -1

        def on_bytes(done: int, total: int | None) -> None:
            nonlocal last
            if progress is None:
                return
            total = total or size
            pct = min(100, int(done * 100 / total)) if total > 0 else 0
            if pct != last:  # 1% ごとに知らせる（ジョブの stage は DB に書くので回数を抑える）
                last = pct
                progress(pct / 100, stage)

        log.info("モデルをダウンロードします: %s → %s", url, path)
        if progress is not None:
            progress(0.0, stage)
        try:
            try:
                self.downloader(url, tmp, on_bytes)
            except (OSError, ValueError) as e:  # 通信・書き込みの失敗（URLError も OSError）
                raise RuntimeError(
                    f"モデル {path.name} をダウンロードできませんでした（{url}）: {e}"
                ) from e
            actual = tmp.stat().st_size if tmp.is_file() else -1
            if actual != size:
                raise RuntimeError(
                    f"ダウンロードしたモデル {path.name} のサイズが違います"
                    f"（{actual} バイト、正しくは {size} バイト）。"
                    "消しました。もう一度実行してください。"
                )
            if sha256_of(tmp) != sha256:
                raise RuntimeError(
                    f"ダウンロードしたモデル {path.name} の SHA-256 が違います（壊れているか、"
                    "配布元のファイルが変わりました）。消しました。もう一度実行してください。"
                )
            tmp.replace(path)
        finally:
            # 失敗・キャンセル（ジョブの中断は downloader の中の進み具合から例外で来る）でも残さない
            tmp.unlink(missing_ok=True)

    def _separate_msst(
        self,
        wav_path: Path,
        model_filename: str,
        options: Mapping[str, Any],
        device: str,
        role: str,
    ) -> dict[str, np.ndarray]:
        import torch

        from stemapp.separation.msst import runner

        if role != ROLE_REFINE:
            raise ValueError(f"{model_filename} は詳細分割（refine）にだけ使えます。")
        if device == DEVICE_CUDA and not torch.cuda.is_available():
            raise RuntimeError("CUDA が使えません（--cpu で CPU 実行できます）。")
        table = REFINE_OUTPUT_NAME_MAP[model_filename]
        ckpt, config = self.ensure_msst_files(model_filename)
        mix = read_audio(wav_path)
        n = mix.shape[0]
        loaded = None
        try:
            loaded = runner.load_model(ckpt, config, list(table), device)
            outputs = runner.separate(
                loaded,
                mix,
                chunk_scale=float(options.get(OPT_CHUNK_SCALE, 1.0)),
                use_fp16=self.use_fp16,
            )
            mapped = map_outputs(role, outputs, model_filename)
            return {k: _fit(v, n) for k, v in mapped.items()}
        finally:
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.shape[0] >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - x.shape[0], x.shape[1]), dtype=np.float32)])
