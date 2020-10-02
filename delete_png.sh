#!/bin/bash
d="$1"
[ "$d" == "" ] && { echo "Usage: $0 directory"; exit 1; }
[ -d "${d}" ] &&  find $1 -type f -name '*.png' -delete || echo "Directory $d not found."

