use rspirv::binary::Assemble;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opt {
    filename: PathBuf,
    #[structopt(short, long, default_value = "a.spv")]
    output: PathBuf,
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

    while spirv_extra_opt_passes::all_passes(&mut module) {}

    let bytes = assemble_module_into_bytes(&module);

    std::fs::write(&opt.output, &bytes).unwrap();
}
