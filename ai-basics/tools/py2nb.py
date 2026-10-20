#!/usr/bin/env python3
"""src/*.py (percent 형식)를 notebooks/*.ipynb로 변환한다.

형식:
    # %% [markdown]      ← 다음 줄부터 마크다운 셀 (각 줄 앞의 "# " 제거)
    # %%                 ← 다음 줄부터 코드 셀

사용법:
    python3 tools/py2nb.py            # src/ 전체 → notebooks/
    python3 tools/py2nb.py src/05_perceptron.py

노트북은 git 비교가 어렵기 때문에 원본은 src/의 .py로 관리하고, 수정 후 이 스크립트로 다시 만든다.
"""
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent


def convert(src: Path, out_dir: Path) -> Path:
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "md":
            lines = [ln[2:] if ln.startswith("# ") else ln[1:] if ln.startswith("#") else ln for ln in text.split("\n")]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in src.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# %% [markdown]"):
            flush(); kind, buf = "md", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        elif kind is not None:
            buf.append(line)
    flush()

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    out = out_dir / (src.stem + ".ipynb")
    nbformat.write(nb, out)
    return out


def main() -> int:
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    srcs = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "src").glob("[0-9][0-9]_*.py"))
    for s in srcs:
        print("→", convert(s, out_dir).relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
