import os
import argparse
from PIL import Image


def create_gif(input_dir, output_file, duration):
    """Create a GIF from images in the specified directory."""
    filenames = [f for f in os.listdir(input_dir)]
    filenames = [f for f in filenames if os.path.isfile(os.path.join(input_dir, f))]
    filenames.sort()
    images = []

    for filename in filenames:
        filepath = os.path.join(input_dir, filename)
        try:
            img = Image.open(filepath)
            images.append(img)
        except:
            print(f"Warning: {filename} could not be opened as an image.")

    if len(images) == 0:
        print("No valid images found in the directory.")
        return

    # Create the GIF
    images[0].save(output_file, save_all=True, append_images=images[1:], loop=0, duration=duration)
    print(f"GIF saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF from images in a directory.")
    parser.add_argument("dir", type=str, help="Directory containing the images.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.gif",
        help="Output GIF file name. Default is 'output.gif'.",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=100,
        help="Duration for each frame in milliseconds. Default is 100ms.",
    )

    args = parser.parse_args()
    create_gif(args.dir, args.output, args.duration)
