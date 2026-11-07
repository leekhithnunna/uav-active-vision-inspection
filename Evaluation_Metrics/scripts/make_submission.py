"""Copies exactly the files to upload into Evaluation_Metrics/SUBMISSION/ and prints their sizes.
Contents: the document (.pdf, .docx), the final MP4s V1-V10, and video_index.md. Anything over 25 MB is flagged."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EM, REPORT, VIDEOS  # noqa: E402

LIMIT_MB = 25.0


def main():
    sub = EM / "SUBMISSION"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir()
    files = [REPORT / "Team08_Evaluation_Metrics.pdf", REPORT / "Team08_Evaluation_Metrics.docx"]
    files += sorted(VIDEOS.glob("V*.mp4"), key=lambda p: int(p.stem.split("_")[0][1:]))
    files += [VIDEOS / "video_index.md"]
    missing = [f for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing deliverables: {missing}")
    total, flagged = 0.0, []
    print(f"{'file':52s} {'size':>9s}")
    for f in files:
        shutil.copy2(f, sub / f.name)
        mb = f.stat().st_size / 1e6
        total += mb
        flag = "  <-- OVER 25 MB" if mb > LIMIT_MB else ""
        flagged += [f.name] if flag else []
        print(f"{f.name:52s} {mb:7.2f} MB{flag}")
    print(f"{len(files)} files, {total:.1f} MB in total -> {sub}")
    print("files over 25 MB: " + (", ".join(flagged) if flagged else "none"))


if __name__ == "__main__":
    main()
