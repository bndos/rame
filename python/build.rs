fn main() {
    pyo3_build_config::use_pyo3_cfgs();

    if std::env::var("RUSTFLAGS")
        .unwrap_or_default()
        .contains("-Cprofile-use=")
    {
        println!("cargo:rustc-cfg=specified_profile_use");
    }
    println!("cargo:rustc-check-cfg=cfg(specified_profile_use)");
    println!(
        "cargo:rustc-env=PROFILE={}",
        std::env::var("PROFILE").unwrap()
    );

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-headerpad_max_install_names");
    }
}
