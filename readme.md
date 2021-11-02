# About

This is an set of experimental optimisation passes for SPIR-V modules. It comes with a binary, `spirv-extra-opt`, and a rust crate, `spirv-extra-opt-passes`.

They are designed to augment the output from [`spirv-opt`](https://github.com/KhronosGroup/SPIRV-Tools), not to replace it. 

A disclaimer: I _really_ wouldn't recommend using this yet. It's pretty likely that you'll end up with broken shaders. If you do use it and encounter an issue, I'd love if it you make a github issue and uploaded the module you ran it on.

```
spirv-extra-opt 0.1.0

USAGE:
    spirv-extra-opt [OPTIONS] <filename>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -o, --output <output>     [default: a.spv]

ARGS:
    <filename> 
```

# Passes

## Vectorisation Pass


The pass I've mostly worked on is the vectorisation one. It looks for a usage pattern where all the components are extracted via `OpCompositeExtract` and then the same operation is ran on each component, e.g. `OpFMul {component} {some scalar}` or `OpFMul {component} {the matching component of a different vector}`.

It then attempts to replace this with the vector form of the operation. E.g. `OpVectorTimesScalar` with a accompanying scalar operand or `OpFMul` with a accompanying vector operand. The follow up instructions are then replaced with `OpCompositeExtract`s that operate on the result of the inserted operation.

The ids of the old instructions are replaced with the new ones wherever they are used as operands. The old instructions are then pruned by the pruning pass.

Additionally, if the components are extracted and then composted into a vector again, then the `OpCompositeConstruct` instruction is just pruned.

This is highly useful for shaders authored by [rust-gpu](https://github.com/EmbarkStudios/rust-gpu).

When this pass is repeated, it can simplify some chains of instructions by quite a lot:

```diff
@@ -755,35 +710,17 @@
         %789 = OpLoad %v3float %788
-        %790 = OpCompositeExtract %float %789 0
-        %791 = OpFMul %float %919 %790
-        %792 = OpCompositeExtract %float %789 1
-        %793 = OpFMul %float %919 %792
-        %794 = OpCompositeExtract %float %789 2
-        %795 = OpFMul %float %919 %794
-        %796 = OpFAdd %float %783 %791
-        %797 = OpFAdd %float %784 %793
-        %798 = OpFAdd %float %785 %795
-        %799 = OpExtInst %float %1 FMax %796 %float_0
-        %800 = OpExtInst %float %1 FMin %799 %float_1
-        %801 = OpExtInst %float %1 FMax %797 %float_0
-        %802 = OpExtInst %float %1 FMin %801 %float_1
-        %803 = OpExtInst %float %1 FMax %798 %float_0
-        %804 = OpExtInst %float %1 FMin %803 %float_1
-        %805 = OpCompositeConstruct %v3float %800 %802 %804
+       %1239 = OpVectorTimesScalar %v3float %789 %919
+       %1252 = OpFAdd %v3float %1234 %1239
+       %1257 = OpExtInst %v3float %1 FMax %1252 %1173
+       %1264 = OpExtInst %v3float %1 FMin %1257 %1269
                OpStore %702 %true
-               OpStore %703 %805
-               OpStore %705 %805
+               OpStore %703 %1264
+               OpStore %705 %1264
```

## Unused Assignment Pruning Pass

This one is a lot simpler. It collects the set of all instruction result ids, all instruction ids that are referenced as types or operands, and computes the difference between the two sets. All unused instructions are then removed.

The only special-casing that it does is for function calls and extension instructions. All the `GLSL.std.450` instructions are assumed to not have any side effects, but isn't the case for other instructions, for example `NonSemantic.DebugPrintf`.

[spirv-tools]: https://github.com/KhronosGroup/SPIRV-Tools
