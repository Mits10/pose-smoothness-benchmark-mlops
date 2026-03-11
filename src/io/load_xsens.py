import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s|%(levelname)s|%(message)s"
)
def process_one_file() -> None:
    return None

def process_directory(input_dir: Path, output_dir: Path, recursive: bool = True) -> None:
    pattern = "**/*.mvnx" if recursive else "*.mvnx"
    files = sorted(input_dir.glob(pattern))

    if not files:
        logging.warning("No mvnx files found in %s", input_dir)
        return
   
    logging.info("Found %d .mvnx files in %s", input_dir)

    for file_path in files:
        try:
            process_one_file(file_path, output_dir)
        except Exception as exc:
            logging.exception("Failed processing %s : %s", file_path,exc)

def main() -> None:

    #Command line Argument
    parser = argparse.ArgumentParser(description="Preprocess MVNX files into wide csv tables.")
    parser.add_argument("--input", required=True, help="Input .mvnx files or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive directory scan")

    #Reading Argument
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    #Validating input exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    #Deciding file vs folder behaviour
    if input_path.is_file():
        if input_path.suffix.lower() != ".mvnx":
            raise ValueError(f"Expected a .mvnx file, got: {input_path}")
        process_one_file(input_path,output_dir)
    else:
        process_directory(input_path,output_dir, recursive = not args.no_recursive)

if __name__ == "__main__":
    main()