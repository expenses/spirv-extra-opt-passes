use rspirv::binary::Assemble;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opt {
    filename: PathBuf,
    #[structopt(short, long, default_value = "a.spv")]
    output: PathBuf,
    #[structopt(long)]
    experimental_remove_bad_op_switches: bool,
    #[structopt(long)]
    normalise_entry_points: bool,
}

// https://github.com/gfx-rs/rspirv/pull/13
fn assemble_module_into_bytes(module: &rspirv::dr::Module) -> Vec<u8> {
    use std::mem;
    module
        .assemble()
        .iter()
        .flat_map(|val| (0..mem::size_of::<u32>()).map(move |i| ((val >> (8 * i)) & 0xff) as u8))
        .collect()
}

fn main() {
    env_logger::init();

    let opt = Opt::from_args();

    let bytes = std::fs::read(&opt.filename).unwrap();

    let mut module = rspirv::dr::load_bytes(&bytes).unwrap();

    loop {
        let mut modified = false;

        modified |= spirv_extra_opt_passes::all_passes(&mut module);
        if opt.experimental_remove_bad_op_switches {
            modified |= spirv_extra_opt_passes::remove_op_switch_with_no_literals(&mut module);
        }

        if !modified {
            break;
        }
    }

    if opt.normalise_entry_points {
        spirv_extra_opt_passes::normalise_entry_points(&mut module);
    }

    let bytes = assemble_module_into_bytes(&module);

    std::fs::write(&opt.output, &bytes).unwrap();
}
