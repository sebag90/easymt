import importlib

from utils.cli import easymt_arguments


def main():
    args = easymt_arguments()
    if args:
        subparser = args.subparser.replace("-", "_")
        module = importlib.import_module(f"bin.{subparser}")
        module.main(args)


if __name__ == "__main__":
    main()
