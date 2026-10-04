import sys

from .cli import main

try:
    sys.exit(main(sys.argv[1:]))
except KeyboardInterrupt:
    sys.exit(130)
