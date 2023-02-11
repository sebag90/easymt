import importlib

from utils.cli import easymt_arguments


def main():
    args = easymt_arguments()
    if args:
        module = importlib.import_module(f"easymt.{args.subparser}")
        module.main(args)


if __name__ == "__main__":
    main()
