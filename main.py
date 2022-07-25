import argparse


def main(args):
    if args.video:
        print("Starting camera...")
    elif args.file:
        print(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the program with file input or camera"
    )

    parser.add_argument(
        "-v",
        "--video",
        help="will attempt to start a camera",
        action="store_true",
    )
    parser.add_argument("-f", "--file", help="use a video file format")

    args = parser.parse_args()

    main(args)
