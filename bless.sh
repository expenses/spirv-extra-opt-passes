#!/bin/sh

cargo run -p spirv-extra-opt-passes $2 -- $1 &&
echo "$(spirv-dis $1 | wc -l) lines to $(spirv-dis a.spv | wc -l) lines" &&
spirv-val a.spv &&
spirv-dis a.spv > test-results/$(basename $1).dis
