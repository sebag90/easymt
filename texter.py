import importlib

from utils.cli import texter_arguments


def main():
    args = texter_arguments()
    if args:
        subparser = args.subparser
        module = importlib.import_module(f"bin.{subparser}")
        module.main(args)


if __name__ == "__main__":
    main()
