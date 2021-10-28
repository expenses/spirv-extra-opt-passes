#!/bin/sh

spirv-dis $1 > _a &&
spirv-dis $2 > _b &&
git diff --no-index _a _b &&
rm _a _b
