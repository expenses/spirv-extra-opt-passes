#!/bin/sh

for file in test-shaders/*.spv; do
./test.sh $file $1 || break
done
