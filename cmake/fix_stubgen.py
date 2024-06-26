#!/usr/bin/env python3

"""
Fixes errors in .pyi files generated by stubgen.
"""

import re
import os
import sys


def main():
    # sys.argv[1] should be a directory in which to search for .pyi files
    filenames = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(sys.argv[1])
        for f in fn
        if f.endswith(".pyi")
    ]
    for filename in filenames:
        with open(filename) as f:
            content = f.read()

        # Convert parameter names from camel case to snake case
        new_content = ""
        extract_location = 0
        for match in re.finditer(r"(?<=Parameter ``)(.*?)(?=``:)", content):
            new_content += content[extract_location : match.start()]
            new_content += re.sub(r"(?<!^)(?=[A-Z])", "_", match.group()).lower()
            extract_location = match.end()
        content = new_content + content[extract_location:]

        with open(filename, mode="w") as f:
            f.write(content)


if __name__ == "__main__":
    main()
