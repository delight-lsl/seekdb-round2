use minivectordb::cli::protocol::run_stdio_loop;
use minivectordb::sql::executor::Database;

fn resolve_data_dir() -> String {
    let mut data_dir = std::env::var("VECTORDB_DATA_DIR").unwrap_or_else(|_| "/data".to_string());
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--data-dir" || arg == "-d" || arg == "--data_dir" {
            if let Some(val) = args.next() {
                if !val.is_empty() {
                    data_dir = val;
                }
            }
        } else if let Some(val) = arg.strip_prefix("--data-dir=") {
            if !val.is_empty() {
                data_dir = val.to_string();
            }
        } else if let Some(val) = arg.strip_prefix("--data_dir=") {
            if !val.is_empty() {
                data_dir = val.to_string();
            }
        } else if arg == "minivectordb" || arg.ends_with("/minivectordb") {
            // Binary name passed as first argument when docker run overrides CMD
            continue;
        } else {
            eprintln!("ignoring unrecognized argument: {}", arg);
        }
    }
    data_dir
}

fn main() {
    let data_dir = resolve_data_dir();
    let mut db = match Database::open(&data_dir) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("failed to open database at '{}': {}", data_dir, e);
            std::process::exit(1);
        }
    };
    if let Err(e) = run_stdio_loop(&mut db) {
        eprintln!("fatal stdio loop error: {}", e);
        std::process::exit(1);
    }
}
