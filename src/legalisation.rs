use crate::{get_glsl_ext_inst_id, get_id_ref};
use num_traits::FromPrimitive;
use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{GLOp, Op, Word};
use std::collections::{hash_map::Entry, HashMap, HashSet};

/// Deduplicate all `OpTypeVector`s. A SPIR-V module is not valid if multiple `OpTypeVector`s
/// are specified with the same scalar type and dimensions.
pub fn dedup_vector_types_pass(module: &mut Module) {
    let mut ty_and_dimensions_to_vector_id = HashMap::new();
    let mut replacements = HashMap::new();

    for instruction in &mut module.types_global_values {
        if instruction.class.opcode != Op::TypeVector {
            continue;
        }

        let scalar = match instruction.operands[0] {
            Operand::IdRef(id) => id,
            _ => continue,
        };

        let dimensions = match instruction.operands[1] {
            Operand::LiteralInt32(dimensions) => dimensions,
            _ => continue,
        };

        let result_id = match instruction.result_id {
            Some(result_id) => result_id,
            _ => continue,
        };

        match ty_and_dimensions_to_vector_id.entry((scalar, dimensions)) {
            Entry::Occupied(matching_vector) => {
                replacements.insert(result_id, *matching_vector.get());
            }
            Entry::Vacant(vacancy) => {
                vacancy.insert(result_id);
            }
        }
    }

    crate::replace_globals(module, &replacements)
}

/// Deduplicate all `OpTypeFunction`s. A SPIR-V module is not valid if multiple `OpTypeFunction`s
/// are specified with the same return type and parameters.
pub fn dedup_type_functions_pass(module: &mut Module) {
    let mut operands_to_function_id = HashMap::new();
    let mut replacements = HashMap::new();

    for instruction in &mut module.types_global_values {
        if instruction.class.opcode != Op::TypeFunction {
            continue;
        }

        let result_id = match instruction.result_id {
            Some(result_id) => result_id,
            _ => continue,
        };

        let operands = instruction
            .operands
            .iter()
            .map(|operand| match operand {
                Operand::IdRef(id) => id,
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();

        match operands_to_function_id.entry(operands) {
            Entry::Occupied(matching_function) => {
                replacements.insert(result_id, *matching_function.get());
            }
            Entry::Vacant(vacancy) => {
                vacancy.insert(result_id);
            }
        }
    }

    crate::replace_globals(module, &replacements)
}

// Some glsl extension functions take scalars even when returning vectors, so we need to make sure we don't switch them to taking vectors
// in the `fix_non_vector_constant_operand` pass. This function returns the index of the scalar operand if the instruction takes one.
fn is_glsl_function_that_takes_scalar(
    instruction: &Instruction,
    glsl_ext_inst_id: Option<Word>,
) -> Option<usize> {
    if gl_op_for_instruction(instruction, glsl_ext_inst_id)? == GLOp::Refract {
        Some(4)
    } else {
        None
    }
}

fn gl_op_for_instruction(
    instruction: &Instruction,
    glsl_ext_inst_id: Option<Word>,
) -> Option<GLOp> {
    let glsl_ext_inst_id = glsl_ext_inst_id?;

    if instruction.class.opcode != Op::ExtInst {
        return None;
    }

    let ext_inst_id = get_id_ref(&instruction.operands[0])?;

    if ext_inst_id != glsl_ext_inst_id {
        return None;
    }

    match &instruction.operands[1] {
        Operand::LiteralExtInstInteger(int) => GLOp::from_u32(*int),
        _ => None,
    }
}

/// Change the operands for specific functions that return vectors from scalars to vectors.
///
/// This needs to happen after the vectorisation pass as passing in scalar operands to certain vector functions it not allowed.
///
/// This might result in multiple OpVector types with the same scalar type and dimensions, so the `dedup_vector_types` pass is ran after this.
pub fn fix_non_vector_operands_pass(module: &mut Module) {
    let constants = module
        .types_global_values
        .iter()
        .filter_map(|instruction| {
            if matches!(
                instruction.class.opcode,
                Op::Constant | Op::ConstantTrue | Op::ConstantFalse
            ) {
                match (instruction.result_id, instruction.result_type) {
                    (Some(result_id), Some(result_type)) => Some((result_id, result_type)),
                    _ => None,
                }
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();

    let scalar_types = module
        .types_global_values
        .iter()
        .filter_map(|instruction| match instruction.class.opcode {
            Op::TypeBool | Op::TypeInt | Op::TypeFloat => instruction.result_id,
            _ => None,
        })
        .collect::<HashSet<_>>();

    let mut id_to_result_scalar_type = HashMap::new();

    let vector_types = module
        .types_global_values
        .iter()
        .filter_map(|instruction| {
            if instruction.class.opcode == Op::TypeVector {
                let dimensions = match instruction.operands[1] {
                    Operand::LiteralInt32(dimensions) => dimensions,
                    _ => return None,
                };

                instruction.result_id.map(|id| (id, dimensions))
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();

    let mut next_id = module.header.as_ref().unwrap().bound;

    let glsl_ext_inst_id = get_glsl_ext_inst_id(module);

    let mut instructions_to_insert = Vec::new();

    for function in &mut module.functions {
        for instruction in function.all_inst_iter_mut() {
            let result_type = match instruction.result_type {
                Some(result_type) => result_type,
                _ => continue,
            };

            if let Some(result_id) = instruction.result_id {
                if scalar_types.contains(&result_type) {
                    id_to_result_scalar_type.insert(result_id, result_type);
                }
            }

            match instruction.class.opcode {
                Op::IEqual
                | Op::FOrdEqual
                | Op::Select
                | Op::ExtInst
                | Op::FSub
                | Op::FAdd
                | Op::FUnordNotEqual
                | Op::FOrdLessThan
                | Op::FOrdLessThanEqual
                | Op::FOrdGreaterThan
                | Op::FOrdGreaterThanEqual
                | Op::FDiv
                | Op::IAdd
                | Op::ISub
                | Op::IMul
                | Op::INotEqual
                | Op::UGreaterThanEqual
                | Op::UGreaterThan
                | Op::ULessThan
                | Op::ULessThanEqual
                | Op::LogicalNotEqual => {}
                _ => continue,
            }

            let result_id = match instruction.result_id {
                Some(result_id) => result_id,
                _ => continue,
            };

            let scalar_operand_index =
                is_glsl_function_that_takes_scalar(instruction, glsl_ext_inst_id);

            if let Some(dimensions) = vector_types.get(&result_type).cloned() {
                for (i, operand) in instruction.operands.iter_mut().enumerate() {
                    let id = match operand {
                        Operand::IdRef(id) => id,
                        _ => continue,
                    };

                    if scalar_operand_index == Some(i) {
                        continue;
                    }

                    // If the argument-that-should-be-a-vector is a constant, make a global OpConstantComposite.
                    if let Some(constant_type) = constants.get(id).cloned() {
                        // Insert a new vector type.
                        let type_vector_id = next_id;
                        module.types_global_values.push(Instruction::new(
                            Op::TypeVector,
                            None,
                            Some(type_vector_id),
                            vec![
                                Operand::IdRef(constant_type),
                                Operand::LiteralInt32(dimensions),
                            ],
                        ));
                        next_id += 1;
                        // Insert a global ConstantComposite.
                        let constant_composite_id = next_id;
                        module.types_global_values.push(Instruction::new(
                            Op::ConstantComposite,
                            Some(type_vector_id),
                            Some(constant_composite_id),
                            vec![Operand::IdRef(*id); dimensions as usize],
                        ));
                        next_id += 1;

                        *operand = Operand::IdRef(constant_composite_id);
                    // If the argument-that-should-be-a-vector isn't constant, we need to insert a
                    // OpCompositeConstruct.
                    } else if let Some(scalar_type) = id_to_result_scalar_type.get(id) {
                        // Insert a new vector type.
                        let type_vector_id = next_id;
                        module.types_global_values.push(Instruction::new(
                            Op::TypeVector,
                            None,
                            Some(type_vector_id),
                            vec![
                                Operand::IdRef(*scalar_type),
                                Operand::LiteralInt32(dimensions),
                            ],
                        ));
                        next_id += 1;

                        let composte_construct_id = next_id;

                        instructions_to_insert.push((
                            result_id,
                            Instruction::new(
                                Op::CompositeConstruct,
                                Some(type_vector_id),
                                Some(composte_construct_id),
                                vec![Operand::IdRef(*id); dimensions as usize],
                            ),
                        ));
                        next_id += 1;

                        *operand = Operand::IdRef(composte_construct_id);
                    }
                }
            }
        }
    }

    insert_instructions(module, &instructions_to_insert);

    module.header.as_mut().unwrap().bound = next_id;

    dedup_vector_types_pass(module);
}

fn insert_instructions(module: &mut Module, to_insert: &[(Word, Instruction)]) {
    'outer: for (id_to_insert_at, instruction_to_insert) in to_insert {
        let id_to_insert_at = *id_to_insert_at;
        let instruction_to_insert = instruction_to_insert.clone();

        for function in &mut module.functions {
            for block in &mut function.blocks {
                let position = block
                    .instructions
                    .iter()
                    .position(|inst| inst.result_id == Some(id_to_insert_at));

                if let Some(position) = position {
                    block.instructions.insert(position, instruction_to_insert);
                    continue 'outer;
                }
            }
        }
    }
}

pub fn fix_wrong_selection_merges(module: &mut Module) -> bool {
    let modified = false;

    for function in &mut module.functions {
        let labels_to_block_instructions = crate::collect_labels_to_block_instructions(&function);

        for block in &mut function.blocks {
            let last_inst = block
                .instructions
                .last()
                .expect("block has no instructions")
                .clone();

            if last_inst.class.opcode == Op::BranchConditional {
                let second_last_index = block.instructions.len() - 2;
                let second_last_inst = block
                    .instructions
                    .get_mut(second_last_index)
                    .expect("BranchConditional needs a merge instruction");
                let mut merge_block_label_id = second_last_inst.operands[0].unwrap_id_ref();

                // Could be LoopMerge instead.
                if second_last_inst.class.opcode == Op::SelectionMerge {
                    //dbg!(&last_inst.operands);

                    for branch_op in &last_inst.operands[1..] {
                        let branch_id = branch_op.unwrap_id_ref();

                        if branch_id != merge_block_label_id {
                            // todo: need to follow branch graph until we find the actual merge point.
                            let (_, block_functions) = labels_to_block_instructions
                                .get(&branch_id)
                                .expect("Invalid Branch");

                            let branch_last_inst =
                                block_functions.last().expect("block has no instructions");

                            if branch_last_inst.class.opcode == Op::Branch {
                                merge_block_label_id = branch_last_inst.operands[0].unwrap_id_ref();

                                //modified = true;
                            }
                        }
                    }

                    second_last_inst.operands[0] = Operand::IdRef(merge_block_label_id);
                }
            }
        }
    }

    modified
}

/*

pub fn remove_useless_blocks(module: &mut Module) {
    for function in &mut module.functions {
        let mut to_replace = HashMap::new();

        for block in &function.blocks {
            if block.instructions.len() == 1 {
                let last_inst = block.instructions.last().unwrap();

                if last_inst.class.opcode == Op::Branch {
                    let branch_to_id = last_inst.operands[0].unwrap_id_ref();

                    let label = block.label.as_ref().expect("All blocks have labels");
        let label_id = label.result_id.expect("All labels have IDs");

                    to_replace.insert(label_id, branch_to_id);
                }
            }
        }

        for instruction in function.all_inst_iter_mut() {
            for operand in &mut instruction.operands {
                if let Operand::IdRef(id) = operand {
                    if let Some(&replacement) = to_replace.get(id) {
                        *id = replacement;
                    }
                }
            }
        }
    }
}

pub fn branch_conditional_thing(module: &mut Module) {
    for function in &mut module.functions {
        for block in &mut function.blocks {
            let last_inst = block.instructions.last().expect("blocks cannot have no instructions");

            if last_inst.class.opcode == Op::BranchConditional {
                if let Some(branch_target) = crate::all_items_equal(last_inst.operands.iter().map(|op| op.unwrap_id_ref())) {
                    dbg!(&branch_target);
                    block.instructions.pop();
                    block.instructions.pop();

                    block.instructions.push(Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(branch_target)]));
                }
            }
        }
    }
}
*/
