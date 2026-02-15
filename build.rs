//! Build script for kv-cache-tier.
//!
//! In a production build, this would:
//! 1. Download or locate llama.cpp source
//! 2. Compile it with CUDA support
//! 3. Generate Rust FFI bindings via bindgen
//!
//! For now, it's a placeholder that documents the intended build process.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Future: compile llama.cpp with CUDA support.
    //
    // Steps:
    // 1. Check for CUDA toolkit (nvcc)
    // 2. Clone/download llama.cpp if not present
    // 3. Use cc::Build to compile the C/C++ sources
    // 4. Link against CUDA runtime (cudart, cublas)
    // 5. Generate bindings with bindgen
    //
    // Example (when implemented):
    //
    // ```
    // let cuda_path = std::env::var("CUDA_PATH")
    //     .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    //
    // cc::Build::new()
    //     .cpp(true)
    //     .file("vendor/llama.cpp/llama.cpp")
    //     .file("vendor/llama.cpp/ggml.c")
    //     .file("vendor/llama.cpp/ggml-cuda.cu")
    //     .include("vendor/llama.cpp")
    //     .include(format!("{cuda_path}/include"))
    //     .flag("-std=c++17")
    //     .define("GGML_USE_CUDA", None)
    //     .compile("llama");
    //
    // println!("cargo:rustc-link-search={cuda_path}/lib64");
    // println!("cargo:rustc-link-lib=cudart");
    // println!("cargo:rustc-link-lib=cublas");
    // ```

    #[cfg(feature = "cuda")]
    {
        println!("cargo:warning=CUDA feature enabled — ensure CUDA toolkit is installed");
    }
}
