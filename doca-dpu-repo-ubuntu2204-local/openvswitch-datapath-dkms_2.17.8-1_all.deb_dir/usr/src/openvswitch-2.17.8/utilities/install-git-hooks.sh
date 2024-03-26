#! /usr/bin/env sh

if [ ! -e .git ] || [ ! git --version >/dev/null 2>&1 ]; then
    # Do not deploy anything if this is not a git-based installation
    exit 0
fi

TOPDIR="$(git rev-parse --show-toplevel)"
TARGET="${TOPDIR}/.git/hooks"
SOURCE="${TOPDIR}/utilities/git-hooks"

for hook in "$SOURCE"/*
do
    tgt="${TARGET}/$(basename $hook)"
    if [ -e "$tgt" ]; then
        echo "$tgt already present"
    else
        ln -sfn "$hook" "$tgt"
    fi
done
