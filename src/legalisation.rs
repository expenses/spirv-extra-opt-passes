use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::Op;
use std::collections::{hash_map::Entry, HashMap};

pub fn dedup_vector_types(module: &mut Module) {
    let mut ty_and_dimensions_to_vector_id = HashMap::new();
    let mut replace = HashMap::new();

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
                replace.insert(result_id, *matching_vector.get());
            }
            Entry::Vacant(vacancy) => {
                vacancy.insert(result_id);
            }
        }
    }

    module
        .types_global_values
        .retain(|instruction| match instruction.result_id {
            Some(result_id) => replace.get(&result_id).is_none(),
            _ => true,
        });

    // For things like OpConstantComposite
    for instruction in &mut module.types_global_values {
        if let Some(result_type) = instruction.result_type.as_mut() {
            if let Some(replace) = replace.get(result_type) {
                *result_type = *replace;
            }
        }
    }

    for function in &mut module.functions {
        for block in &mut function.blocks {
            for instruction in &mut block.instructions {
                if let Some(result_type) = instruction.result_type.as_mut() {
                    if let Some(replace) = replace.get(result_type) {
                        *result_type = *replace;
                    }
                }
            }
        }
    }
}

// After the vectorisation pass, some operands need to be changed from constant scalars
// to vector scalars.
pub fn fix_non_vector_constant_operand(module: &mut Module) {
    let constants = module
        .types_global_values
        .iter()
        .filter_map(|instruction| {
            if instruction.class.opcode == Op::Constant {
                match (instruction.result_id, instruction.result_type) {
                    (Some(result_id), Some(result_type)) => Some((result_id, result_type)),
                    _ => None,
                }
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();

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

    for function in &mut module.functions {
        for block in &mut function.blocks {
            for instruction in &mut block.instructions {
                match instruction.class.opcode {
                    Op::IEqual | Op::FOrdEqual | Op::Select | Op::ExtInst => {}
                    _ => continue,
                }

                let result_type = match instruction.result_type {
                    Some(result_type) => result_type,
                    _ => continue,
                };

                if let Some(dimensions) = vector_types.get(&result_type).cloned() {
                    for operand in &mut instruction.operands {
                        let id = match operand {
                            Operand::IdRef(id) => id,
                            _ => continue,
                        };

                        if let Some(constant_type) = constants.get(&id).cloned() {
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
                            let constant_composite_id = next_id;
                            module.types_global_values.push(Instruction::new(
                                Op::ConstantComposite,
                                Some(type_vector_id),
                                Some(constant_composite_id),
                                vec![Operand::IdRef(*id); dimensions as usize],
                            ));
                            next_id += 1;

                            *operand = Operand::IdRef(constant_composite_id);
                        }
                    }
                }
            }
        }
    }

    module.header.as_mut().unwrap().bound = next_id;
}
