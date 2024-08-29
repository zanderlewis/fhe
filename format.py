import black
import os


def format_files():
    # Format python files in all directories and subdirectories
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    code = f.read()
                # Create a default mode
                mode = black.FileMode()
                try:
                    formatted_code = black.format_file_contents(
                        code, fast=True, mode=mode
                    )
                    with open(file_path, "w") as f:
                        f.write(formatted_code)
                except black.NothingChanged:
                    # Ignore the "nothing changed" error
                    pass


if __name__ == "__main__":
    format_files()
