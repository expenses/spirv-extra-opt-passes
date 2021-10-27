use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

pub fn remove_unused_assignments(module: &mut Module) -> bool {
    let mut unused = HashSet::new();

    fn is_forward_referenced(opcode: Op) -> bool {
        match opcode {
            Op::Label | Op::Phi | Op::Function => true,
            _ => false,
        }
    }

    fn handle_instruction(unused: &mut HashSet<Word>, instruction: &Instruction) {
        for operand in &instruction.operands {
            if let Operand::IdRef(id) = operand {
                unused.remove(&id);
            }
        }

        if let Some(result_type) = &instruction.result_type {
            unused.remove(result_type);
        }

        // Forwards referenced types are used before they are assigned, so we ignore them as we're doing a single pass.
        if is_forward_referenced(instruction.class.opcode) {
            return;
        }

        if let Some(result_id) = instruction.result_id {
            unused.insert(result_id);
        }
    }

    for instruction in &module.types_global_values {
        handle_instruction(&mut unused, &instruction);
    }

    for function in &module.functions {
        if let Some(instruction) = &function.def {
            handle_instruction(&mut unused, instruction);
        }

        for block in &function.blocks {
            for instruction in &block.instructions {
                handle_instruction(&mut unused, &instruction);
            }
        }
    }

    for instruction in &module.entry_points {
        handle_instruction(&mut unused, &instruction);
    }

    module
        .types_global_values
        .retain(|instruction| match instruction.result_id {
            Some(result_ty) => !unused.contains(&result_ty),
            _ => true,
        });

    for function in &mut module.functions {
        for block in &mut function.blocks {
            block
                .instructions
                .retain(|instruction| match instruction.result_id {
                    Some(result_ty) => !unused.contains(&result_ty),
                    _ => true,
                });
        }
    }

    let mut removed_debug_name = false;

    module.debug_names.retain(|instruction| {
        let mut has_id_operands = false;
        let mut all_id_operands_are_unused = true;

        for operand in &instruction.operands {
            if let Operand::IdRef(id) = operand {
                has_id_operands = true;
                all_id_operands_are_unused &= unused.contains(&id);
            }
        }

        let remove = has_id_operands && all_id_operands_are_unused;

        removed_debug_name |= remove;

        !remove
    });

    !unused.is_empty() || removed_debug_name
}

pub fn handle_vector_decomposition(module: &mut Module) -> bool {
    #[derive(Debug, Copy, Clone, PartialEq)]
    struct VectorTypeInfo {
        dimensions: u32,
        scalar_id: u32,
        id: u32,
    }

    #[derive(Debug)]
    struct VectorInfo {
        extracted_component_ids: Vec<Option<u32>>,
        ty: VectorTypeInfo,
        id: u32,
    }

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

    let mut changed = false;

    for function in &mut module.functions {
        for block in &mut function.blocks {
            let mut vector_info = HashMap::new();
            let mut follow_up_instructions: HashMap<u32, Option<Instruction>> = HashMap::new();
            let mut instructions_to_replace: HashMap<u32, Instruction> = HashMap::new();
            let mut ids_to_replace: HashMap<u32, u32> = HashMap::new();
            let mut op_composite_extract_inst_to_vector_id = HashMap::new();

            for instruction in &block.instructions {
                for operand in &instruction.operands {
                    if let Operand::IdRef(id) = operand {
                        if let Some(follow_up_instruction) = follow_up_instructions.get_mut(id) {
                            follow_up_instruction.get_or_insert(instruction.clone());
                        }
                    }
                }

                match instruction.class.opcode {
                    Op::Load | Op::VectorTimesScalar => {
                        let (result_id, result_type) =
                            match (instruction.result_id, instruction.result_type) {
                                (Some(result_id), Some(result_type)) => (result_id, result_type),
                                _ => continue,
                            };

                        if let Some(vector_type) = vector_type_info.get(&result_type).cloned() {
                            vector_type_info.insert(result_id, vector_type);
                        }
                    }
                    Op::CompositeExtract => {
                        let vector_id = match instruction.operands[0] {
                            Operand::IdRef(id) => id,
                            _ => continue,
                        };
                        let index = match instruction.operands[1] {
                            Operand::LiteralInt32(index) => index,
                            _ => continue,
                        };
                        let vector_type = match vector_type_info.get(&vector_id) {
                            Some(vector_type) => *vector_type,
                            _ => continue,
                        };

                        let result_id = instruction.result_id.unwrap();

                        vector_type_info.insert(result_id, vector_type);
                        op_composite_extract_inst_to_vector_id.insert(result_id, vector_id);

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

                        vector_info.extracted_component_ids[index as usize] = Some(result_id);

                        follow_up_instructions.insert(result_id, None);
                    }
                    _ => {}
                }
            }

            let mut vector_info = vector_info.iter().collect::<Vec<_>>();
            vector_info.sort_unstable_by_key(|&(id, _)| id);

            'outer: for (_, vector_info) in &vector_info {
                let extracted_component_ids = match vector_info
                    .extracted_component_ids
                    .iter()
                    .cloned()
                    .collect::<Option<Vec<_>>>()
                {
                    Some(extracted_component_ids) => extracted_component_ids,
                    None => continue,
                };

                let followup_instructions = match extracted_component_ids
                    .iter()
                    .map(|id| follow_up_instructions[&id].as_ref())
                    .collect::<Option<Vec<&Instruction>>>()
                {
                    Some(followup_instructions) => followup_instructions,
                    None => continue,
                };

                let opcode = match all_items_equal(
                    followup_instructions.iter().map(|inst| inst.class.opcode),
                ) {
                    Some(opcode) => opcode,
                    None => continue,
                };

                if opcode == Op::CompositeConstruct {
                    continue;
                }

                if opcode == Op::FOrdEqual {
                    continue;
                }

                let vector_opcode = match opcode {
                    Op::FMul => Op::VectorTimesScalar,
                    _ => opcode,
                };

                let shared_first_operand =
                    all_items_equal(followup_instructions.iter().map(|inst| &inst.operands[0]));
                let shared_second_operand =
                    all_items_equal(followup_instructions.iter().map(|inst| &inst.operands[1]));

                let all_followups_point_to_vector =
                    all_items_equal_filter(followup_instructions.iter().map(|inst| {
                        let id = match inst.operands[1] {
                            Operand::IdRef(id) => id,
                            _ => return None,
                        };

                        let id2 = match inst.operands[0] {
                            Operand::IdRef(id) => id,
                            _ => return None,
                        };

                        let first_operand_is_in_block =
                            op_composite_extract_inst_to_vector_id.get(&id2).is_some();

                        if first_operand_is_in_block {
                            op_composite_extract_inst_to_vector_id.get(&id)
                        } else {
                            None
                        }
                    }));

                let new_instruction_operands = shared_first_operand
                    .map(|operand| {
                        if opcode == Op::FMul {
                            vec![Operand::IdRef(vector_info.id), operand.clone()]
                        } else {
                            vec![operand.clone(), Operand::IdRef(vector_info.id)]
                        }
                    })
                    .or(shared_second_operand
                        .map(|operand| vec![Operand::IdRef(vector_info.id), operand.clone()]))
                    .or(all_followups_point_to_vector
                        .map(|id| vec![Operand::IdRef(vector_info.id), Operand::IdRef(*id)]));

                if let Some(new_instruction_operands) = new_instruction_operands {
                    println!(
                        "Replacing {} with {:?}",
                        extracted_component_ids[0], vector_opcode
                    );

                    // Replace the first OpCompositeExtract with the VectorTimes Scalar and the rest with Nops.
                    instructions_to_replace.insert(
                        extracted_component_ids[0],
                        Instruction::new(
                            vector_opcode,
                            Some(vector_info.ty.id),
                            Some(extracted_component_ids[0]),
                            new_instruction_operands,
                        ),
                    );
                    instructions_to_replace.insert(
                        extracted_component_ids[1],
                        Instruction::new(Op::Nop, None, None, vec![]),
                    );
                    instructions_to_replace.insert(
                        extracted_component_ids[2],
                        Instruction::new(Op::Nop, None, None, vec![]),
                    );

                    // As we're moving the extraction until later, we need to replace usages of the extracted IDs with the followup IDs.
                    for i in 0..vector_info.ty.dimensions {
                        ids_to_replace.insert(
                            extracted_component_ids[i as usize],
                            followup_instructions[i as usize].result_id.unwrap(),
                        );
                    }

                    for (i, inst) in followup_instructions.iter().enumerate() {
                        instructions_to_replace.insert(
                            inst.result_id.unwrap(),
                            Instruction::new(
                                Op::CompositeExtract,
                                Some(vector_info.ty.scalar_id),
                                Some(inst.result_id.unwrap()),
                                vec![
                                    Operand::IdRef(extracted_component_ids[0]),
                                    Operand::LiteralInt32(i as u32),
                                ],
                            ),
                        );
                    }

                    break 'outer;
                }
            }

            for instruction in &mut block.instructions {
                let result_id = match instruction.result_id {
                    Some(result_id) => result_id,
                    None => continue,
                };

                if let Some(new_id) = ids_to_replace.get(&result_id) {
                    instruction.result_id = Some(*new_id);
                    changed = true;
                }

                if let Some(new) = instructions_to_replace.get(&result_id) {
                    *instruction = new.clone();
                    changed = true;
                }
            }

            block
                .instructions
                .retain(|instruction| instruction.class.opcode != Op::Nop);
        }
    }

    dbg!(changed)
}

fn all_items_equal_filter<T: PartialEq>(
    mut iterator: impl Iterator<Item = Option<T>>,
) -> Option<T> {
    let first_item = match iterator.next() {
        Some(Some(item)) => item,
        _ => return None,
    };

    for item in iterator {
        if item.as_ref() != Some(&first_item) {
            return None;
        }
    }

    Some(first_item)
}

fn all_items_equal<T: PartialEq>(mut iterator: impl Iterator<Item = T>) -> Option<T> {
    let first_item = match iterator.next() {
        Some(item) => item,
        None => return None,
    };

    for item in iterator {
        if item != first_item {
            return None;
        }
    }

    Some(first_item)
}

pub fn all_passes(module: &mut Module) -> bool {
    handle_vector_decomposition(module) || remove_unused_assignments(module)
}
