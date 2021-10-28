use num_traits::cast::FromPrimitive;
use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

mod legalisation;

use legalisation::{dedup_vector_types, fix_non_vector_constant_operand};

pub fn unused_assignment_pruning_pass(module: &mut Module) -> bool {
    let mut result_ids = HashSet::new();
    let mut referenced_ids = HashSet::new();

    let glsl_ext_inst_id = get_glsl_ext_inst_id(&module);

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

        for block in &function.blocks {
            for instruction in &block.instructions {
                handle_instruction(
                    &mut referenced_ids,
                    &mut result_ids,
                    instruction,
                    glsl_ext_inst_id,
                );
            }
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

    let mut removed_debug_name_or_annotation = false;

    module.debug_names.retain(|instruction| {
        let mut has_id_operands = false;
        let mut all_id_operands_are_unused = true;

        for operand in &instruction.operands {
            if let Operand::IdRef(id) = operand {
                has_id_operands = true;
                all_id_operands_are_unused &= unused.contains(id);
            }
        }

        let remove = has_id_operands && all_id_operands_are_unused;

        removed_debug_name_or_annotation |= remove;

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

        removed_debug_name_or_annotation |= remove;

        !remove
    });

    !unused.is_empty() || removed_debug_name_or_annotation
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

struct CompositeExtractInfo {
    vector_id: Word,
    dimension_index: u32,
}

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

    let glsl_ext_inst_id = get_glsl_ext_inst_id(&module);

    let mut next_id = module.header.as_ref().unwrap().bound;

    let mut changed = false;

    let mut ids_to_replace: HashMap<u32, u32> = HashMap::new();

    for function in &mut module.functions {
        for block in &mut function.blocks {
            let mut vector_info = HashMap::new();
            let mut follow_up_instructions: HashMap<u32, Option<FollowUpInstruction>> =
                HashMap::new();
            let mut composite_extract_info = HashMap::new();

            for (insertion_index, instruction) in block.instructions.iter().enumerate() {
                for operand in &instruction.operands {
                    if let Operand::IdRef(id) = operand {
                        if let Some(follow_up_instruction) = follow_up_instructions.get_mut(id) {
                            follow_up_instruction.get_or_insert(FollowUpInstruction {
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

                        follow_up_instructions.insert(result_id, None);
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

            for (_, vector_info) in &vector_info {
                let modified = vectorise(
                    vector_info,
                    &follow_up_instructions,
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
                    break;
                }
            }
        }
    }

    for function in &mut module.functions {
        for block in &mut function.blocks {
            for instruction in &mut block.instructions {
                for operand in &mut instruction.operands {
                    let id = match operand {
                        Operand::IdRef(id) => id,
                        _ => continue,
                    };

                    if let Some(new_id) = ids_to_replace.get(&id) {
                        *operand = Operand::IdRef(*new_id);
                    }
                }
            }

            block
                .instructions
                .retain(|instruction| instruction.class.opcode != Op::Nop);
        }
    }

    module.header.as_mut().unwrap().bound = next_id;

    if changed {
        fix_non_vector_constant_operand(module);
        dedup_vector_types(module);
        unused_assignment_pruning_pass(module);
    }

    changed
}

fn vectorise(
    vector_info: &VectorInfo,
    follow_up_instructions: &HashMap<Word, Option<FollowUpInstruction>>,
    composite_extract_info: &HashMap<Word, CompositeExtractInfo>,
    ids_to_replace: &mut HashMap<Word, Word>,
    types_global_values: &mut Vec<Instruction>,
    instructions: &mut Vec<Instruction>,
    next_id: &mut u32,
    glsl_ext_inst_id: Option<Word>,
) -> Option<()> {
    let extracted_component_ids = vector_info
        .extracted_component_ids
        .iter()
        .cloned()
        .collect::<Option<Vec<_>>>()?;

    // We want to insert the instruction at the index of the first follow up instruction.
    //
    // Inserting sooner, like at the first component extract instruction, might not work as
    // any other operands might not be defined.
    let insertion_index = follow_up_instructions
        .get(&extracted_component_ids[0])?
        .as_ref()?
        .insertion_index;

    let follow_up_instructions = extracted_component_ids
        .iter()
        .map(|id| {
            follow_up_instructions[id]
                .as_ref()
                .map(|follow_up| &follow_up.instruction)
        })
        .collect::<Option<Vec<&Instruction>>>()?;

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

            let is_not_swizzled = same_instruction
                .operands
                .iter()
                .zip(extracted_component_ids)
                .all(|(operand, component_id)| match operand {
                    Operand::IdRef(id) => *id == component_id,
                    _ => false,
                });

            if !is_not_swizzled {
                return None;
            }

            let composite_construct_id = follow_up_instruction_result_ids[0];

            println!(
                "Removing references to {:?} at {}",
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

    // Only allow certain types for now
    match opcode {
        Op::FMul
        | Op::FAdd
        | Op::FSub
        | Op::IEqual
        | Op::IAdd
        | Op::FOrdEqual
        | Op::Select
        | Op::ExtInst => {}
        _ => {
            println!("!!Unhandled opcode: {:?}!!", opcode);
            return None;
        }
    }

    let operands = get_operands(
        vector_info.id,
        &follow_up_instructions,
        composite_extract_info,
        &mut opcode,
        glsl_ext_inst_id,
    )?;

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

    println!(
        "Replacing {} with {:?} ({})",
        extracted_component_ids[0], opcode, inserted_instruction_id
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

fn vector_operand(
    operand_index: usize,
    follow_up_instructions: &[&Instruction],
    composite_extract_info: &HashMap<Word, CompositeExtractInfo>,
) -> Option<Word> {
    all_items_equal_filter(follow_up_instructions.iter().enumerate().map(|(i, inst)| {
        let id = match inst.operands[operand_index] {
            Operand::IdRef(id) => id,
            _ => return None,
        };

        match composite_extract_info.get(&id) {
            Some(&CompositeExtractInfo {
                vector_id,
                dimension_index,
            }) if dimension_index == i as u32 => Some(vector_id),
            _ => None,
        }
    }))
}

fn glsl_operands<'a>(
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

fn get_operands(
    vector_id: Word,
    follow_up_instructions: &[&Instruction],
    composite_extract_info: &HashMap<Word, CompositeExtractInfo>,
    opcode: &mut Op,
    glsl_ext_inst_id: Option<Word>,
) -> Option<Vec<Operand>> {
    if *opcode == Op::ExtInst {
        let glsl_ext_inst_id = glsl_ext_inst_id?;

        let (gl_op, other_operand) = glsl_operands(follow_up_instructions, glsl_ext_inst_id, 2)
            .or(glsl_operands(follow_up_instructions, glsl_ext_inst_id, 3))?;

        // todo: only some gl ops can be vectorised.
        let gl_op = match gl_op {
            Operand::LiteralExtInstInteger(int) => rspirv::spirv::GLOp::from_u32(*int)?,
            _ => return None,
        };

        return Some(vec![
            Operand::IdRef(glsl_ext_inst_id),
            Operand::LiteralExtInstInteger(gl_op as u32),
            Operand::IdRef(vector_id),
            other_operand.clone(),
        ]);
    }

    if *opcode == Op::Select {
        let (true_op, false_op) = all_items_equal(
            follow_up_instructions
                .iter()
                .map(|inst| (&inst.operands[1], &inst.operands[2])),
        )?;

        return Some(vec![
            Operand::IdRef(vector_id),
            true_op.clone(),
            false_op.clone(),
        ]);
    }

    let shared_first_operand =
        all_items_equal(follow_up_instructions.iter().map(|inst| &inst.operands[0]));
    let shared_second_operand =
        all_items_equal(follow_up_instructions.iter().map(|inst| &inst.operands[1]));

    let vector_first_operand = vector_operand(0, &follow_up_instructions, composite_extract_info);

    let vector_second_operand = vector_operand(1, &follow_up_instructions, composite_extract_info);

    let (other_operand, vector_is_first_operand, other_operand_is_vector) = shared_first_operand
        .map(|operand| {
            // The operation will get turned into OpVectorTimesScalar which needs the vector to be in the first position.
            let vector_is_first = *opcode == Op::FMul;
            (operand.clone(), vector_is_first, false)
        })
        .or_else(|| shared_second_operand.map(|operand| (operand.clone(), true, false)))
        .or_else(|| vector_first_operand.map(|id| (Operand::IdRef(id), false, true)))
        .or_else(|| vector_second_operand.map(|id| (Operand::IdRef(id), true, true)))?;

    if *opcode == Op::FMul && !other_operand_is_vector {
        *opcode = Op::VectorTimesScalar;
    }

    Some(if vector_is_first_operand {
        vec![Operand::IdRef(vector_id), other_operand]
    } else {
        vec![other_operand, Operand::IdRef(vector_id)]
    })
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

pub fn all_passes(module: &mut Module) -> bool {
    let mut modified = vectorisation_pass(module);

    while unused_assignment_pruning_pass(module) {
        modified = true;
    }

    modified
}
