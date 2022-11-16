use rspirv::dr::{Function, Instruction, Module, Operand};
use rspirv::spirv::{Op, Word};
use std::collections::{hash_map::Entry, HashMap, HashSet};

pub mod legalisation;
mod vectorisation_operands;

use legalisation::{dedup_type_functions_pass, fix_non_vector_operands_pass};
use vectorisation_operands::get_operands;

fn has_side_effects(opcode: Op) -> bool {
    match opcode {
        Op::FunctionCall => true,
        Op::ReportIntersectionKHR => true,
        // todo: there are probably a bunch more.
        _ => false,
    }
}

/// See 'Unused Assignment Pruning Pass' in readme.md
pub fn unused_assignment_pruning_pass(module: &mut Module) -> bool {
    let mut result_ids = HashSet::new();
    let mut referenced_ids = HashSet::new();

    let glsl_ext_inst_id = get_glsl_ext_inst_id(module);

    fn is_unknown_extension_instruction(
        instruction: &Instruction,
        glsl_ext_inst_id: Option<Word>,
    ) -> bool {
        if instruction.class.opcode != Op::ExtInst {
            return false;
        }

        let glsl_ext_inst_id = match glsl_ext_inst_id {
            Some(id) => id,
            _ => return true,
        };

        match instruction.operands[0] {
            Operand::IdRef(id) => id != glsl_ext_inst_id,
            _ => true,
        }
    }

    fn handle_instruction(
        referenced_ids: &mut HashSet<Word>,
        result_ids: &mut HashSet<Word>,
        instruction: &Instruction,
        glsl_ext_inst_id: Option<Word>,
    ) {
        for operand in &instruction.operands {
            if let Operand::IdRef(id) = operand {
                referenced_ids.insert(*id);
            }
        }

        if let Some(result_type) = &instruction.result_type {
            referenced_ids.insert(*result_type);
        }

        // Function calls have side effects so we don't prune them even if the return type isn't used (which is common).
        if has_side_effects(instruction.class.opcode) {
            return;
        }

        if is_unknown_extension_instruction(instruction, glsl_ext_inst_id) {
            return;
        }

        if let Some(result_id) = instruction.result_id {
            result_ids.insert(result_id);
        }
    }

    for instruction in &module.types_global_values {
        handle_instruction(
            &mut referenced_ids,
            &mut result_ids,
            instruction,
            glsl_ext_inst_id,
        );
    }

    for function in &module.functions {
        if let Some(instruction) = &function.def {
            handle_instruction(
                &mut referenced_ids,
                &mut result_ids,
                instruction,
                glsl_ext_inst_id,
            );
        }

        for instruction in &function.parameters {
            handle_instruction(
                &mut referenced_ids,
                &mut result_ids,
                instruction,
                glsl_ext_inst_id,
            );
        }

        let mut passed_first_label = false;

        for instruction in function.all_inst_iter() {
            // The first instruction (the function label) is not referenced by anything so we shouldn't DCE it.
            if !passed_first_label && instruction.class.opcode == Op::Label {
                passed_first_label = true;
                continue;
            }

            handle_instruction(
                &mut referenced_ids,
                &mut result_ids,
                instruction,
                glsl_ext_inst_id,
            );
        }
    }

    for instruction in &module.entry_points {
        handle_instruction(
            &mut referenced_ids,
            &mut result_ids,
            instruction,
            glsl_ext_inst_id,
        );
    }

    let unused = result_ids
        .difference(&referenced_ids)
        .collect::<HashSet<_>>();

    let mut removed_instruction = false;

    module.types_global_values.retain(|instruction| {
        let remove = match instruction.result_id {
            Some(result_ty) => unused.contains(&result_ty),
            _ => false,
        };

        removed_instruction |= remove;

        !remove
    });

    for function in &mut module.functions {
        function.blocks.retain(|block| {
            let label = block.label.as_ref().expect("All blocks have labels");
            let label_id = label.result_id.expect("All labels have IDs");

            let remove = unused.contains(&label_id);

            removed_instruction |= remove;

            !remove
        });

        for block in &mut function.blocks {
            block.instructions.retain(|instruction| {
                let remove = match instruction.result_id {
                    Some(result_ty) => unused.contains(&result_ty),
                    _ => false,
                };

                removed_instruction |= remove;

                !remove
            });
        }
    }

    module.functions.retain(|function| {
        let remove = match &function.def {
            Some(instruction) => match instruction.result_id {
                Some(id) => unused.contains(&id),
                None => false,
            },
            None => false,
        };

        removed_instruction |= remove;

        !remove
    });

    // Keep debug names only if the id of the debug name is not known.
    // This is a little different from the rest of the `retain`s where we use `unused`.
    // That's because we don't treat debug names as a 'source' of referenced ids in the first place
    // so the id doesn't show up as being unused.
    module.debug_names.retain(|instruction| {
        let mut has_id_operands = false;
        let mut all_id_operands_are_unused = true;

        for operand in &instruction.operands {
            if let Operand::IdRef(id) = operand {
                has_id_operands = true;
                all_id_operands_are_unused &= !result_ids.contains(id);
            }
        }

        let remove = has_id_operands && all_id_operands_are_unused;

        removed_instruction |= remove;

        !remove
    });

    module.annotations.retain(|instruction| {
        let mut has_id_operands = false;
        let mut all_id_operands_are_unused = true;

        for operand in &instruction.operands {
            if let Operand::IdRef(id) = operand {
                has_id_operands = true;
                all_id_operands_are_unused &= unused.contains(id);
            }
        }

        let remove = has_id_operands && all_id_operands_are_unused;

        removed_instruction |= remove;

        !remove
    });

    let removed_unused_function_param = removed_unused_function_params(module, &unused);

    removed_instruction || removed_unused_function_param
}

// `unused` can contain function params. We can't just remove these though without changing the
// corresponding `OpTypeFunction` and `OpFunctionCall`s.
fn removed_unused_function_params(module: &mut Module, unused: &HashSet<&Word>) -> bool {
    fn remove_sorted_indices_with_offset<T>(
        vec: &mut Vec<T>,
        sorted_indices: &[usize],
        offset: usize,
    ) {
        for i in sorted_indices.iter().rev() {
            vec.remove(*i + offset);
        }
    }

    let mut function_params_to_remove = HashMap::new();
    let mut next_id = module.header.as_ref().unwrap().bound;

    for function in &mut module.functions {
        let mut params_to_remove = Vec::new();

        let (function_id, function_return_ty) = match &function.def {
            Some(inst) => match (inst.result_id, inst.result_type) {
                (Some(result_id), Some(result_type)) => (result_id, result_type),
                _ => continue,
            },
            _ => continue,
        };

        for (i, instruction) in function.parameters.iter().enumerate() {
            if let Some(result_id) = instruction.result_id {
                if unused.contains(&result_id) {
                    params_to_remove.push(i);
                }
            }
        }

        if !params_to_remove.is_empty() {
            remove_sorted_indices_with_offset(&mut function.parameters, &params_to_remove, 0);

            function_params_to_remove.insert(function_id, params_to_remove);

            let function_type_id = next_id;

            // Push a new OpTypeFunction with the new operands and set it as the type for the function.
            module.types_global_values.push(Instruction::new(
                Op::TypeFunction,
                None,
                Some(function_type_id),
                {
                    std::iter::once(Operand::IdRef(function_return_ty))
                        .chain(
                            function
                                .parameters
                                .iter()
                                .map(|inst| match inst.result_type {
                                    Some(id) => Operand::IdRef(id),
                                    None => unreachable!(),
                                }),
                        )
                        .collect::<Vec<_>>()
                },
            ));
            next_id += 1;

            if let Some(inst) = &mut function.def {
                inst.operands[1] = Operand::IdRef(function_type_id);
            }
        }
    }

    for instruction in module.all_inst_iter_mut() {
        if instruction.class.opcode == Op::FunctionCall {
            let function_id = match &instruction.operands[0] {
                Operand::IdRef(id) => id,
                _ => continue,
            };

            if let Some(to_remove) = function_params_to_remove.get(&function_id) {
                // Remove the unused function parameters. We use an offset of 1 because the
                // function being called is the first operand.
                remove_sorted_indices_with_offset(&mut instruction.operands, to_remove, 1);
            }
        }
    }

    let modified = !function_params_to_remove.is_empty();

    if modified {
        // Dedup the OpTypeFunctions as having duplicates with the same operands is not allowed.
        dedup_type_functions_pass(module);
    }

    module.header.as_mut().unwrap().bound = next_id;

    modified
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct VectorTypeInfo {
    dimensions: u32,
    scalar_id: Word,
    id: Word,
}

#[derive(Debug)]
struct VectorInfo {
    extracted_component_ids: Vec<Option<Word>>,
    ty: VectorTypeInfo,
    id: Word,
}

#[derive(Debug)]
struct CompositeExtractInfo {
    vector_id: Word,
    dimension_index: u32,
    num_dimensions: u32,
}

#[derive(Debug)]
struct FollowUpInstruction {
    instruction: Instruction,
    insertion_index: usize,
}

fn get_glsl_ext_inst_id(module: &Module) -> Option<Word> {
    let mut glsl_ext_inst_id = None;

    for instruction in &module.ext_inst_imports {
        if instruction.class.opcode == Op::ExtInstImport {
            if let Operand::LiteralString(lit_string) = &instruction.operands[0] {
                if lit_string == "GLSL.std.450" {
                    glsl_ext_inst_id = instruction.result_id;
                }
            }
        }
    }

    glsl_ext_inst_id
}

/// See 'Vectorisation Pass' in readme.md
pub fn vectorisation_pass(module: &mut Module) -> bool {
    let mut vector_type_info = HashMap::new();

    for instruction in &module.types_global_values {
        if instruction.class.opcode == Op::TypeVector {
            if let Some(id) = instruction.result_id {
                vector_type_info.insert(
                    id,
                    VectorTypeInfo {
                        id,
                        scalar_id: match instruction.operands[0] {
                            Operand::IdRef(id) => id,
                            _ => continue,
                        },
                        dimensions: match instruction.operands[1] {
                            Operand::LiteralInt32(dimensions) => dimensions,
                            _ => continue,
                        },
                    },
                );
            }
        }
    }

    let glsl_ext_inst_id = get_glsl_ext_inst_id(module);

    let mut next_id = module.header.as_ref().unwrap().bound;

    let mut changed = false;

    let mut ids_to_replace: HashMap<u32, u32> = HashMap::new();
    let mut composite_extract_info = HashMap::new();

    for function in &mut module.functions {
        for block in &mut function.blocks {
            let mut vector_info = HashMap::new();
            let mut follow_up_instructions: HashMap<u32, Vec<FollowUpInstruction>> = HashMap::new();

            for (insertion_index, instruction) in block.instructions.iter().enumerate() {
                for operand in &instruction.operands {
                    if let Operand::IdRef(id) = operand {
                        if let Some(follow_up_instruction) = follow_up_instructions.get_mut(id) {
                            follow_up_instruction.push(FollowUpInstruction {
                                instruction: instruction.clone(),
                                insertion_index,
                            });
                        }
                    }
                }

                match instruction.class.opcode {
                    Op::CompositeExtract => {
                        let (vector_id, dimension_index) =
                            match (&instruction.operands[0], &instruction.operands[1]) {
                                (
                                    &Operand::IdRef(vector_id),
                                    &Operand::LiteralInt32(dimension_index),
                                ) => (vector_id, dimension_index),
                                _ => continue,
                            };

                        let vector_type = match vector_type_info.get(&vector_id) {
                            Some(vector_type) => *vector_type,
                            _ => continue,
                        };

                        let result_id = instruction.result_id.unwrap();

                        vector_type_info.insert(result_id, vector_type);
                        composite_extract_info.insert(
                            result_id,
                            CompositeExtractInfo {
                                vector_id,
                                dimension_index,
                                num_dimensions: vector_type.dimensions,
                            },
                        );

                        let vector_info =
                            vector_info.entry(vector_id).or_insert_with(|| VectorInfo {
                                extracted_component_ids: match vector_type.dimensions {
                                    2 => vec![None; 2],
                                    3 => vec![None; 3],
                                    4 => vec![None; 4],
                                    _ => unreachable!(),
                                },
                                ty: vector_type,
                                id: vector_id,
                            });

                        vector_info.extracted_component_ids[dimension_index as usize] =
                            Some(result_id);

                        follow_up_instructions.insert(result_id, Vec::new());
                    }
                    _ => {
                        let (result_id, result_type) =
                            match (instruction.result_id, instruction.result_type) {
                                (Some(result_id), Some(result_type)) => (result_id, result_type),
                                _ => continue,
                            };

                        if let Some(vector_type) = vector_type_info.get(&result_type).cloned() {
                            vector_type_info.insert(result_id, vector_type);
                        }
                    }
                }
            }

            let mut vector_info = vector_info.iter().collect::<Vec<_>>();
            vector_info.sort_unstable_by_key(|&(id, _)| id);

            'vector_info: for (_, vector_info) in &vector_info {
                let extracted_component_ids = match vector_info
                    .extracted_component_ids
                    .iter()
                    .cloned()
                    .collect::<Option<Vec<_>>>()
                {
                    Some(component_ids) => component_ids,
                    None => continue,
                };

                let first_follow_ups = match follow_up_instructions.get(&extracted_component_ids[0])
                {
                    Some(follow_ups) => follow_ups,
                    _ => continue,
                };

                'follow_ups: for follow_up in first_follow_ups {
                    let opcode = follow_up.instruction.class.opcode;

                    let mut instructions = vec![&follow_up.instruction];
                    let mut insertion_index = follow_up.insertion_index;

                    // Find all the other follow up instructions that share the same opcode.
                    for component_id in &extracted_component_ids[1..] {
                        let other_follow_up = match follow_up_instructions
                            .get(component_id)
                            .and_then(|follow_ups| {
                                follow_ups
                                    .iter()
                                    .find(|follow_up| follow_up.instruction.class.opcode == opcode)
                            }) {
                            Some(follow_up) => follow_up,
                            _ => continue 'follow_ups,
                        };

                        instructions.push(&other_follow_up.instruction);
                        // We want to insert the instruction at the index of the first follow up instruction.
                        //
                        // Inserting sooner, like at the first component extract instruction, might not work as
                        // any other operands might not be defined.
                        insertion_index = insertion_index.min(other_follow_up.insertion_index);
                    }

                    let modified = vectorise(
                        vector_info,
                        &extracted_component_ids,
                        &instructions,
                        insertion_index,
                        &composite_extract_info,
                        &mut ids_to_replace,
                        &mut module.types_global_values,
                        &mut block.instructions,
                        &mut next_id,
                        glsl_ext_inst_id,
                    )
                    .is_some();

                    changed |= modified;

                    if modified {
                        // Break from the loop as it's easier to just re-run the pass than update then state for a second
                        // modification.
                        break 'vector_info;
                    }
                }
            }
        }
    }

    for function in &mut module.functions {
        for instruction in function.all_inst_iter_mut() {
            for operand in &mut instruction.operands {
                let id = match operand {
                    Operand::IdRef(id) => id,
                    _ => continue,
                };

                if let Some(new_id) = ids_to_replace.get(id) {
                    *operand = Operand::IdRef(*new_id);
                }
            }
        }
    }

    module.header.as_mut().unwrap().bound = next_id;

    if changed {
        fix_non_vector_operands_pass(module);
        unused_assignment_pruning_pass(module);
    }

    changed
}

fn vectorise(
    vector_info: &VectorInfo,
    extracted_component_ids: &[Word],
    follow_up_instructions: &[&Instruction],
    insertion_index: usize,
    composite_extract_info: &HashMap<Word, CompositeExtractInfo>,
    ids_to_replace: &mut HashMap<Word, Word>,
    types_global_values: &mut Vec<Instruction>,
    instructions: &mut Vec<Instruction>,
    next_id: &mut u32,
    glsl_ext_inst_id: Option<Word>,
) -> Option<()> {
    let scalar_type = follow_up_instructions[0].result_type?;

    let follow_up_instruction_result_ids = follow_up_instructions
        .iter()
        .map(|instruction| instruction.result_id)
        .collect::<Option<Vec<_>>>()?;

    let same_instruction =
        all_items_equal_by_key(follow_up_instructions.iter(), |inst| inst.class.opcode)?;

    let mut opcode = same_instruction.class.opcode;

    if all_items_equal(follow_up_instruction_result_ids.iter()).is_some() {
        // In the event that all the components end up being composited again,
        // we check if there isn't any swizzling happening and if not just replace
        // the id and leave it to be pruned.
        if opcode == Op::CompositeConstruct {
            if same_instruction.operands.len() != extracted_component_ids.len() {
                return None;
            }

            let composite_type = scalar_type;

            // If the composite isn't the same type as the vector being deconstructed (it could be a struct with the same fields)
            // then we can't do anything.
            if vector_info.ty.id != composite_type {
                return None;
            }

            let is_not_swizzled = same_instruction
                .operands
                .iter()
                .zip(extracted_component_ids)
                .all(|(operand, component_id)| match operand {
                    Operand::IdRef(id) => id == component_id,
                    _ => false,
                });

            if !is_not_swizzled {
                return None;
            }

            let composite_construct_id = follow_up_instruction_result_ids[0];

            log::trace!(
                "Removing references to {:?} at %{}",
                Op::CompositeConstruct,
                composite_construct_id
            );

            ids_to_replace.insert(composite_construct_id, vector_info.id);

            return Some(());
        } else {
            // There shouldn't be any other vectorisable cases where all 3 follow up instructions are the same.
            return None;
        }
    }

    // Only allow certain opcodes for now
    match opcode {
        Op::FMul
        | Op::FAdd
        | Op::FSub
        | Op::FDiv
        | Op::IEqual
        | Op::IAdd
        | Op::FOrdEqual
        | Op::FUnordNotEqual
        | Op::FOrdGreaterThan
        | Op::FOrdGreaterThanEqual
        | Op::FOrdLessThan
        | Op::FOrdLessThanEqual
        | Op::Select
        | Op::ExtInst
        | Op::FNegate
        | Op::ConvertSToF
        | Op::ConvertUToF
        | Op::ISub
        | Op::IMul
        | Op::INotEqual
        | Op::UGreaterThanEqual
        | Op::UGreaterThan
        | Op::ULessThan
        | Op::ULessThanEqual
        | Op::LogicalNotEqual
        | Op::FConvert => {}
        // todo: spirv-val thinks that bitcasting vectors is legal, but I'm not sure if it's correct.
        Op::Bitcast => {}
        // Don't think we can handle the case where each component is used as the scalar in a vector times scalar op.
        Op::VectorTimesScalar => return None,
        Op::CompositeConstruct => return None,
        _ => {
            log::debug!(target: "unhandled", "Unhandled opcode: {:?}", opcode);
            return None;
        }
    }

    let operands = match get_operands(
        vector_info,
        follow_up_instructions,
        composite_extract_info,
        &mut opcode,
        glsl_ext_inst_id,
    ) {
        Some(operands) => operands,
        None => {
            log::debug!(
                "Failed to get operands for {:?} at %{}",
                opcode,
                follow_up_instruction_result_ids[0]
            );

            return None;
        }
    };

    let vector_id = *next_id;

    types_global_values.push(Instruction::new(
        Op::TypeVector,
        None,
        Some(vector_id),
        vec![
            Operand::IdRef(scalar_type),
            Operand::LiteralInt32(vector_info.ty.dimensions),
        ],
    ));

    *next_id += 1;

    let inserted_instruction_id = *next_id;

    log::trace!(
        "Replacing %{} with {:?} (%{})",
        extracted_component_ids[0],
        opcode,
        inserted_instruction_id
    );

    // Replace the first OpCompositeExtract with the VectorTimes Scalar and the rest with Nops.
    instructions.insert(
        insertion_index,
        Instruction::new(
            opcode,
            Some(vector_id),
            Some(inserted_instruction_id),
            operands,
        ),
    );

    *next_id += 1;

    for (i, result_id) in follow_up_instruction_result_ids.iter().cloned().enumerate() {
        instructions.insert(
            insertion_index + i + 1,
            Instruction::new(
                Op::CompositeExtract,
                Some(scalar_type),
                Some(*next_id),
                vec![
                    Operand::IdRef(inserted_instruction_id),
                    Operand::LiteralInt32(i as u32),
                ],
            ),
        );

        ids_to_replace.insert(result_id, *next_id);

        *next_id += 1;
    }

    Some(())
}

pub fn all_passes(module: &mut Module) -> bool {
    let mut modified = vectorisation_pass(module);

    while unused_assignment_pruning_pass(module) {
        modified = true;
    }

    modified |= dedup_constant_composites_pass(module);

    modified
}

pub fn dedup_constant_composites_pass(module: &mut Module) -> bool {
    let mut ty_and_operand_to_composite_id = HashMap::new();
    let mut replacements = HashMap::new();

    for instruction in &mut module.types_global_values {
        if instruction.class.opcode != Op::ConstantComposite {
            continue;
        }

        let result_id = match instruction.result_id {
            Some(result_id) => result_id,
            _ => continue,
        };

        let result_type = match instruction.result_type {
            Some(result_type) => result_type,
            _ => continue,
        };

        let operands = match instruction
            .operands
            .iter()
            .map(|operand| match operand {
                Operand::IdRef(id) => Some(id),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            Some(operands) => operands,
            _ => continue,
        };

        match ty_and_operand_to_composite_id.entry((result_type, operands)) {
            Entry::Occupied(matching_composite) => {
                replacements.insert(result_id, *matching_composite.get());
            }
            Entry::Vacant(vacancy) => {
                vacancy.insert(result_id);
            }
        }
    }

    replace_globals(module, &replacements);

    !replacements.is_empty()
}

fn collect_labels_to_block_instructions(
    function: &Function,
) -> HashMap<Word, (usize, Vec<Instruction>)> {
    let mut labels_to_block_instructions = HashMap::new();

    for (block_index, block) in function.blocks.iter().enumerate() {
        let label = block.label.as_ref().expect("All blocks have labels");
        let label_id = label.result_id.expect("All labels have IDs");

        labels_to_block_instructions.insert(label_id, (block_index, block.instructions.clone()));
    }

    labels_to_block_instructions
}

// Inline blocks that follow a `OpSwitch <OpConstant 0> _` that rust-gpu/spirv-opt likes to insert.
// This compiles to a `switch(0)` in webgl glsl which seems to break some shaders such as ones that use discard.
//
// This pass is a work in progress and breaks some test shaders. It requires `fix_wrong_selection_merges` to work
// properly in order to be enabled by default.
pub fn remove_op_switch_with_no_literals(module: &mut Module) -> bool {
    let mut modified = false;
    let instruction_reference_counts = count_instruction_references_in_operands(module, |_| true);
    for function in &mut module.functions {
        let labels_to_block_instructions = collect_labels_to_block_instructions(&function);

        let mut blocks_to_remove = HashSet::new();

        for block in &mut function.blocks {
            let last_inst = block
                .instructions
                .last()
                .expect("blocks cannot have no instructions");

            if last_inst.class.opcode == Op::Switch && last_inst.operands.len() == 2 {
                let label_id = last_inst.operands[1].unwrap_id_ref();

                if let Some(&num_references) = instruction_reference_counts.get(&label_id) {
                    // We append blocks only if they are referenced a single time as otherwise we're just duplicating code.
                    if num_references < 2 {
                        let (block_to_merge_index, instructions_to_append) =
                            labels_to_block_instructions
                                .get(&label_id)
                                .expect("Invalid switch");

                        block.instructions.pop();
                        // For the SelectionMerge.
                        block.instructions.pop();

                        block
                            .instructions
                            .extend_from_slice(&instructions_to_append);

                        blocks_to_remove.insert(block_to_merge_index);

                        modified = true;
                    }
                }
            }
        }

        // We remove unrealable blocks in DCE too, but that code breaks when IDs are doubled-up,
        // so we make sure to remove the blocks here beforehabd.

        let mut block_id = 0;

        function.blocks.retain(|_| {
            let remove = blocks_to_remove.contains(&block_id);
            block_id += 1;
            !remove
        });
    }

    modified |= legalisation::fix_wrong_selection_merges(module);

    modified
}

fn replace_globals(module: &mut Module, replacements: &HashMap<Word, Word>) {
    module
        .types_global_values
        .retain(|instruction| match instruction.result_id {
            Some(result_id) => replacements.get(&result_id).is_none(),
            _ => true,
        });

    for instruction in &mut module.types_global_values {
        if let Some(result_type) = instruction.result_type.as_mut() {
            if let Some(replacement) = replacements.get(result_type) {
                *result_type = *replacement;
            }
        }
    }

    for function in &mut module.functions {
        if let Some(def) = &mut function.def {
            if let Operand::IdRef(function_type_id) = &mut def.operands[1] {
                if let Some(replacement) = replacements.get(function_type_id) {
                    *function_type_id = *replacement;
                }
            }
        }

        for instruction in function.all_inst_iter_mut() {
            if let Some(result_type) = instruction.result_type.as_mut() {
                if let Some(replacement) = replacements.get(result_type) {
                    *result_type = *replacement;
                }
            }

            for operand in &mut instruction.operands {
                let id = match operand {
                    Operand::IdRef(id) => id,
                    _ => continue,
                };

                if let Some(replacement) = replacements.get(id) {
                    *operand = Operand::IdRef(*replacement);
                }
            }
        }
    }
}

fn all_items_equal<T: PartialEq>(mut iterator: impl Iterator<Item = T>) -> Option<T> {
    let first_item = iterator.next()?;

    for item in iterator {
        if item != first_item {
            return None;
        }
    }

    Some(first_item)
}

fn all_items_equal_by_key<I, T: PartialEq>(
    mut iterator: impl Iterator<Item = I>,
    closure: impl Fn(&I) -> T,
) -> Option<I> {
    let first_item = match iterator.next() {
        Some(item) => item,
        None => return None,
    };

    let first_key = closure(&first_item);

    for item in iterator {
        if closure(&item) != first_key {
            return None;
        }
    }

    Some(first_item)
}

fn all_items_equal_filter<T: PartialEq>(
    mut iterator: impl Iterator<Item = Option<T>>,
) -> Option<T> {
    let first_item = iterator.next()??;

    for item in iterator {
        if item.as_ref() != Some(&first_item) {
            return None;
        }
    }

    Some(first_item)
}

fn get_id_ref(operand: &Operand) -> Option<Word> {
    match operand {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    }
}

fn count_instruction_references_in_operands<F: Fn(&Instruction) -> bool>(
    module: &Module,
    filter: F,
) -> HashMap<Word, u32> {
    let mut counts = HashMap::new();

    for function in &module.functions {
        for instruction in function.all_inst_iter() {
            if filter(instruction) {
                for operand in &instruction.operands {
                    if let &Operand::IdRef(id) = operand {
                        *counts.entry(id).or_default() += 1;
                    }
                }
            }
        }
    }

    counts
}

pub fn normalise_entry_points(module: &mut Module) {
    for entry_point in &mut module.entry_points {
        let name = &mut entry_point.operands[2];
        let new_name = name.unwrap_literal_string().replace(':', "_");
        *name = Operand::LiteralString(new_name);
    }
}
