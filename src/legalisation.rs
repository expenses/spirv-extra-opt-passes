use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

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

        if let Some(existing) = ty_and_dimensions_to_vector_id.get(&(scalar, dimensions)) {
            replace.insert(result_id, *existing);
        } else {
            ty_and_dimensions_to_vector_id.insert((scalar, dimensions), result_id);
        }
    }

    module
        .types_global_values
        .retain(|instruction| match instruction.result_id {
            Some(result_id) => replace.get(&result_id).is_none(),
            _ => true,
        });

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
