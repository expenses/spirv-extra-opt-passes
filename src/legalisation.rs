use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::Op;
use std::collections::{
    hash_map::Entry,
    HashMap,
    //HashSet
};

/// Deduplicate all OpTypeVectors. A SPIR-V module is not valid if multiple OpTypeVectors
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

/// Change the operands for specific functions that return vectors from constant scalars to constant vectors.
///
/// This needs to happen after the vectorisation pass as passing in scalar operands to certain vector functions it not allowed.
///
/// This might result in multiple OpVector types with the same scalar type and dimensions, so the `dedup_vector_types` pass should be ran after this.
pub fn fix_non_vector_constant_operand(module: &mut Module) {
    let constants = module
        .types_global_values
        .iter()
        .filter_map(|instruction| {
            if matches!(instruction.class.opcode, Op::Constant | Op::ConstantFalse) {
                match (instruction.result_id, instruction.result_type) {
                    (Some(result_id), Some(result_type)) => Some((result_id, result_type)),
                    _ => None,
                }
            } else {
                None
            }
        })
        .collect::<HashMap<_, _>>();

    /*
    let scalar_types = module.types_global_values.iter()
        .filter_map(|instruction| {
            match instruction.class.opcode {
                Op::TypeBool | Op::TypeInt | Op::TypeFloat => instruction.result_id,
                _ => None
            }
        })
        .collect::<HashSet<_>>();

    let mut id_to_result_scalar_type = HashMap::new();
    */

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
                let result_type = match instruction.result_type {
                    Some(result_type) => result_type,
                    _ => continue,
                };

                /*
                if let Some(result_id) = instruction.result_id {
                    if scalar_types.contains(&result_type) {
                        id_to_result_scalar_type.insert(result_id, result_type);
                    }
                }
                */

                match instruction.class.opcode {
                    Op::IEqual
                    | Op::FOrdEqual
                    | Op::Select
                    | Op::ExtInst
                    | Op::FSub
                    | Op::FAdd
                    | Op::FUnordNotEqual
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
                    | Op::ULessThanEqual => {}
                    _ => continue,
                }

                if let Some(dimensions) = vector_types.get(&result_type).cloned() {
                    for operand in &mut instruction.operands {
                        let id = match operand {
                            Operand::IdRef(id) => id,
                            _ => continue,
                        };

                        if let Some(constant_type) = constants.get(id).cloned() {
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
                        } else {
                            // todo: might need to handle this case.
                            // if let Some(scalar_type) = id_to_result_scalar_type.get(id) {}
                        }
                    }
                }
            }
        }
    }

    module.header.as_mut().unwrap().bound = next_id;
}
