#!/bin/sh

for file in test-shaders/*.spv; do
./test.sh $file
done
