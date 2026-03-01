from typing import List
import utils as u

import testing as t


def main():

    docs_contents, docs_names, docs_paths = u.read_docs()
    t.test_chunking(docs_contents, docs_names)


if __name__ == '__main__':
    main()


