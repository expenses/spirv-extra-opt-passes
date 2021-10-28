#!/bin/sh

for file in test-shaders/*.spv; do
./bless.sh $file $1 || break
done
