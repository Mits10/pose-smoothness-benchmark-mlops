from pathlib import Path


def main() -> None:
    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitkeep").touch(exist_ok=True)
    print("Prepared interim dataset directory.")


if __name__ == "__main__":
    main()