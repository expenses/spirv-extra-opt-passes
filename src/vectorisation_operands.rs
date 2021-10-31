use crate::{all_items_equal, all_items_equal_filter, CompositeExtractInfo, VectorInfo};
use num_traits::cast::FromPrimitive;
use rspirv::dr::{Instruction, Operand};
use rspirv::spirv::{GLOp, Op, Word};
use std::collections::HashMap;

fn get_id_ref(operand: &Operand) -> Option<Word> {
    match operand {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    }
}

fn shared_vector_operand_at_index(
    vector_info: &VectorInfo,
    operand_index: usize,
    follow_up_instructions: &[&Instruction],
    composite_extract_info: &HashMap<Word, CompositeExtractInfo>,
) -> Option<Word> {
    all_items_equal_filter(follow_up_instructions.iter().enumerate().map(|(i, inst)| {
        let id = get_id_ref(&inst.operands[operand_index])?;

        match composite_extract_info.get(&id) {
            Some(&CompositeExtractInfo {
                vector_id: other_vector_id,
                dimension_index,
                num_dimensions: other_num_dimensions,
            }) if dimension_index == i as u32
                && other_vector_id != vector_info.id
                && other_num_dimensions == vector_info.ty.dimensions =>
            {
                Some(other_vector_id)
            }
            _ => None,
        }
    }))
}

fn takes_vector_twice(
    vector_id: Word,
    follow_up_instructions: &[&Instruction],
    composite_extract_info: &HashMap<Word, CompositeExtractInfo>,
) -> bool {
    follow_up_instructions.iter().enumerate().all(|(i, inst)| {
        let id_1 = match inst.operands[0] {
            Operand::IdRef(id) => id,
            _ => return false,
        };

        let id_2 = match inst.operands[1] {
            Operand::IdRef(id) => id,
            _ => return false,
        };

        let i = i as u32;

        match (
            composite_extract_info.get(&id_1),
            composite_extract_info.get(&id_2),
        ) {
            (
                Some(&CompositeExtractInfo {
                    vector_id: vector_id_1,
                    dimension_index: dimension_index_1,
                    ..
                }),
                Some(&CompositeExtractInfo {
                    vector_id: vector_id_2,
                    dimension_index: dimension_index_2,
                    ..
                }),
            ) => {
                dimension_index_1 == i
                    && dimension_index_2 == i
                    && vector_id_1 == vector_id
                    && vector_id_2 == vector_id
            }
            _ => false,
        }
    })
}

fn operands_for_two_operand_glsl_inst<'a>(
    follow_up_instructions: &[&'a Instruction],
    glsl_ext_inst_id: Word,
    other_index: usize,
) -> Option<(&'a Operand, &'a Operand)> {
    all_items_equal_filter(follow_up_instructions.iter().map(|inst| {
        if inst.operands.len() != 4 {
            return None;
        }

        if inst.operands[0] != Operand::IdRef(glsl_ext_inst_id) {
            return None;
        }

        Some((&inst.operands[1], &inst.operands[other_index]))
    }))
}

pub(crate) fn get_operands(
    vector_info: &VectorInfo,
    follow_up_instructions: &[&Instruction],
    composite_extract_info: &HashMap<Word, CompositeExtractInfo>,
    opcode: &mut Op,
    glsl_ext_inst_id: Option<Word>,
) -> Option<Vec<Operand>> {
    if *opcode == Op::ExtInst {
        let glsl_ext_inst_id = glsl_ext_inst_id?;

        let (gl_op, other_operand) =
            operands_for_two_operand_glsl_inst(follow_up_instructions, glsl_ext_inst_id, 2)
                .or_else(|| {
                    operands_for_two_operand_glsl_inst(follow_up_instructions, glsl_ext_inst_id, 3)
                })?;

        let gl_op = match gl_op {
            Operand::LiteralExtInstInteger(int) => GLOp::from_u32(*int)?,
            _ => return None,
        };

        // todo: it's possible that some scalar glsl ops can't be vectorised. More testing is needed.

        return Some(vec![
            Operand::IdRef(glsl_ext_inst_id),
            Operand::LiteralExtInstInteger(gl_op as u32),
            Operand::IdRef(vector_info.id),
            other_operand.clone(),
        ]);
    }

    if *opcode == Op::Select {
        let false_op = all_items_equal(follow_up_instructions.iter().map(|inst| &inst.operands[2]))
            .cloned()?;

        let true_op = shared_vector_operand_at_index(
            vector_info,
            1,
            follow_up_instructions,
            composite_extract_info,
        )
        .map(|id| Operand::IdRef(id))
        .or_else(|| {
            all_items_equal(follow_up_instructions.iter().map(|inst| &inst.operands[1])).cloned()
        })?;

        return Some(vec![
            Operand::IdRef(vector_info.id),
            true_op,
            false_op.clone(),
        ]);
    }

    // Single operand instructions such as FNegate are easily converted into vector instructions.
    if follow_up_instructions
        .iter()
        .all(|inst| inst.operands.len() == 1)
    {
        return Some(vec![Operand::IdRef(vector_info.id)]);
    }

    let shared_first_operand =
        all_items_equal(follow_up_instructions.iter().map(|inst| &inst.operands[0]));
    let shared_second_operand =
        all_items_equal(follow_up_instructions.iter().map(|inst| &inst.operands[1]));

    let vector_first_operand = shared_vector_operand_at_index(
        vector_info,
        0,
        follow_up_instructions,
        composite_extract_info,
    );

    let vector_second_operand = shared_vector_operand_at_index(
        vector_info,
        1,
        follow_up_instructions,
        composite_extract_info,
    );

    let takes_vector_twice = takes_vector_twice(
        vector_info.id,
        follow_up_instructions,
        composite_extract_info,
    );

    let (other_operand, vector_is_first_operand, other_operand_is_vector) = shared_first_operand
        .map(|operand| {
            // The operation will get turned into OpVectorTimesScalar which needs the vector to be in the first position.
            let vector_is_first = *opcode == Op::FMul;
            (operand.clone(), vector_is_first, false)
        })
        .or_else(|| shared_second_operand.map(|operand| (operand.clone(), true, false)))
        .or_else(|| vector_first_operand.map(|id| (Operand::IdRef(id), false, true)))
        .or_else(|| vector_second_operand.map(|id| (Operand::IdRef(id), true, true)))
        .or_else(|| {
            if takes_vector_twice {
                Some((Operand::IdRef(vector_info.id), true, true))
            } else {
                None
            }
        })?;

    if *opcode == Op::FMul && !other_operand_is_vector {
        *opcode = Op::VectorTimesScalar;
    }

    Some(if vector_is_first_operand {
        vec![Operand::IdRef(vector_info.id), other_operand]
    } else {
        vec![other_operand, Operand::IdRef(vector_info.id)]
    })
}
