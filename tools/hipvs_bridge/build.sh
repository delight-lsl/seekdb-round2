#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT=${1:-$SCRIPT_DIR/libseekdb_cuvs_bridge.so}
mkdir -p "$(dirname "$OUTPUT")"

HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
HIPCC=${HIPCC:-$ROCM_ROOT/bin/hipcc}

if [[ -x "$HIPCC" && -f "$HIPVS_ROOT/include/cuvs/core/c_api.h" ]]; then
  echo "Building hipVS bridge for AMD ROCm via hipcc..."
  ARCH_FLAGS=()
  if [[ -x "$ROCM_ROOT/bin/rocm_agent_enumerator" ]]; then
    while IFS= read -r arch; do
      [[ -n "$arch" && "$arch" != "gfx000" ]] && ARCH_FLAGS+=("--offload-arch=$arch")
    done < <("$ROCM_ROOT/bin/rocm_agent_enumerator" 2>/dev/null | grep -E '^gfx[0-9a-z]+' | sort -u || true)
  fi
  if [[ ${#ARCH_FLAGS[@]} -eq 0 ]]; then
    ARCH_FLAGS=("--offload-arch=gfx1100" "--offload-arch=gfx90a" "--offload-arch=gfx942")
  fi
  EXTRA_INC=()
  while IFS= read -r hp; do
    inc_dir=${hp%/raft/core/host_mdspan.hpp}
    [[ -d "$inc_dir" ]] && EXTRA_INC+=("-I$inc_dir")
  done < <(find "$HIPVS_ROOT" "$ROCM_ROOT" -path "*/raft/core/host_mdspan.hpp" 2>/dev/null | head -n 4 || true)

  CPP_LIBS=(-lcuvs_c)
  if [[ -f "$HIPVS_ROOT/lib/libcuvs.so" || -f "$HIPVS_ROOT/lib64/libcuvs.so" ]]; then
    CPP_LIBS+=(-lcuvs)
  elif [[ -f "$HIPVS_ROOT/lib/libhipvs.so" || -f "$HIPVS_ROOT/lib64/libhipvs.so" ]]; then
    CPP_LIBS+=(-lhipvs)
  fi

  if "$HIPCC" -x hip -shared -fPIC -O3 -std=c++17 "${ARCH_FLAGS[@]}" \
      -D__HIP_PLATFORM_AMD__ -DUSE_CUVS_CPP_OPTIMIZE=1 -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
      -I"$HIPVS_ROOT/include" -I"$ROCM_ROOT/include" "${EXTRA_INC[@]}" \
      "$SCRIPT_DIR/seekdb_cuvs_bridge.cu" \
      -L"$HIPVS_ROOT/lib" -L"$HIPVS_ROOT/lib64" -L"$ROCM_ROOT/lib" \
      -Wl,-soname,libseekdb_cuvs_bridge.so \
      -Wl,-rpath,"$HIPVS_ROOT/lib:$HIPVS_ROOT/lib64:$ROCM_ROOT/lib" \
      "${CPP_LIBS[@]}" -lpthread -o "$OUTPUT"; then
    echo "Enabled hipVS C++ CAGRA fast-path."
  else
    echo "Falling back to hipVS C API bridge build..."
    "$HIPCC" \
      -x hip \
      -shared \
      -fPIC \
      -O3 \
      -std=c++17 \
      "${ARCH_FLAGS[@]}" \
      -D__HIP_PLATFORM_AMD__ \
      -I"$HIPVS_ROOT/include" \
      -I"$ROCM_ROOT/include" \
      "$SCRIPT_DIR/seekdb_cuvs_bridge.cu" \
      -L"$HIPVS_ROOT/lib" \
      -L"$HIPVS_ROOT/lib64" \
      -L"$ROCM_ROOT/lib" \
      -Wl,-soname,libseekdb_cuvs_bridge.so \
      -Wl,-rpath,"$HIPVS_ROOT/lib:$HIPVS_ROOT/lib64:$ROCM_ROOT/lib" \
      -lcuvs_c \
      -lpthread \
      -o "$OUTPUT"
  fi
elif command -v nvcc >/dev/null 2>&1 || [[ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
  echo "Building cuVS bridge for NVIDIA CUDA via nvcc..."
  CUDA_ROOT=${CUDA_HOME:-/usr/local/cuda}
  NVCC_BIN=$(command -v nvcc 2>/dev/null || echo "$CUDA_ROOT/bin/nvcc")
  CUVS_ROOT=${CUVS_ROOT:-/usr/local}
  CCBIN_ARGS=()
  if [[ -n "${CXX:-}" ]]; then
    CCBIN_ARGS=(-ccbin "$CXX" -allow-unsupported-compiler -Xcompiler "-fuse-ld=lld -Wno-unused-command-line-argument")
  elif ! command -v g++ >/dev/null 2>&1 || ! g++ -E -x c++ - </dev/null >/dev/null 2>&1; then
    if command -v clang++ >/dev/null 2>&1; then
      CCBIN_ARGS=(-ccbin "$(command -v clang++)" -allow-unsupported-compiler -Xcompiler "-fuse-ld=lld -Wno-unused-command-line-argument")
    fi
  fi
  if "$NVCC_BIN" \
      "${CCBIN_ARGS[@]}" \
      -arch=native \
      --expt-relaxed-constexpr \
      -shared \
      -Xcompiler "-fPIC -O3" \
      -O3 \
      -std=c++17 \
      -DUSE_CUVS_CPP_OPTIMIZE=1 \
      -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE \
      -I"$CUVS_ROOT/include" \
      -I"$CUDA_ROOT/include" \
      "$SCRIPT_DIR/seekdb_cuvs_bridge.cu" \
      -L"$CUDA_ROOT/lib64" \
      -L"$CUVS_ROOT/lib" \
      -Xlinker -soname -Xlinker libseekdb_cuvs_bridge.so \
      -Xlinker --allow-shlib-undefined \
      -Xlinker -rpath -Xlinker "$CUDA_ROOT/lib64:$CUVS_ROOT/lib" \
      -lcudart \
      -lcuvs_c \
      -lcuvs \
      -lpthread \
      -o "$OUTPUT"; then
    echo "Enabled cuVS C++ CAGRA fast-path."
  else
    "$NVCC_BIN" \
      "${CCBIN_ARGS[@]}" \
      -arch=native \
      -shared \
      -Xcompiler "-fPIC -O3" \
      -O3 \
      -std=c++17 \
      -I"$CUVS_ROOT/include" \
      -I"$CUDA_ROOT/include" \
      "$SCRIPT_DIR/seekdb_cuvs_bridge.cu" \
      -L"$CUDA_ROOT/lib64" \
      -L"$CUVS_ROOT/lib" \
      -Xlinker -soname -Xlinker libseekdb_cuvs_bridge.so \
      -Xlinker --allow-shlib-undefined \
      -Xlinker -rpath -Xlinker "$CUDA_ROOT/lib64:$CUVS_ROOT/lib" \
      -lcudart \
      -lcuvs_c \
      -lpthread \
      -o "$OUTPUT"
  fi
else
  echo "Error: Neither hipcc (ROCm) nor nvcc (CUDA) environment found." >&2
  exit 2
fi

echo "built $OUTPUT"
