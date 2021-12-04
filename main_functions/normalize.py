import os
from pathlib import Path
import shutil

from utils.utils import name_suffix_from_file


def normalize(args):

    full_name = args.file.split(os.sep)[-1]
    name, suffix = name_suffix_from_file(full_name)

    if args.subword is True:
        # undo bpe splitting
        os.system(f"sed -r 's/(@@ )|(@@ ?$)//g' < {args.file}> data/temp.txt")
    else:
        shutil.copyfile(Path(args.file), "data/temp.txt")

    # detokenize
    script = Path(f"preprocessing-tools/detokenizer.perl")
    infile = Path(f"data/temp.txt")
    ofile = Path(f"data/{name}.normalized.{suffix}")
    os.system(
        f"perl {script} -u -l {suffix} < {infile} > {ofile}"
    )
