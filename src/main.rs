use minivectordb::cli::protocol::run_stdio_loop;
use minivectordb::sql::executor::Database;

fn resolve_data_dir() -> Result<String, String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--data-dir" {
            let val = args
                .next()
                .ok_or_else(|| "missing path after --data-dir".to_string())?;
            return Ok(val);
        } else if let Some(val) = arg.strip_prefix("--data-dir=") {
            if val.is_empty() {
                return Err("empty path in --data-dir=".to_string());
            }
            return Ok(val.to_string());
        } else {
            return Err(format!("unknown argument: {}", arg));
        }
    }
    Ok(std::env::var("VECTORDB_DATA_DIR").unwrap_or_else(|_| "/data".to_string()))
}

fn main() {
    let data_dir = match resolve_data_dir() {
        Ok(d) => d,
        Err(msg) => {
            eprintln!("argument error: {}", msg);
            std::process::exit(2);
        }
    };
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
