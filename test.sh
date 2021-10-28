#!/bin/sh

cargo run -p spirv-extra-opt-passes -- $1 &&
spirv-val a.spv &&
echo "$(spirv-dis $1 | wc -l) lines to $(spirv-dis a.spv | wc -l) lines" &&
spirv-dis a.spv > a.dis &&
git diff --no-index test-results/$(basename $1).dis a.dis
