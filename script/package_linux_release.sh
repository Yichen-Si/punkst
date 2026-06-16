#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  script/package_linux_release.sh --version VERSION --tier TIER --glibc-min VERSION [options]

Build and package a native Linux punkst release tarball without Docker.

Required:
  --version VERSION       Release version, for example v0.1.0
  --tier TIER             CPU tier: x86_64, x86_64-v3, or x86_64-v4
  --glibc-min VERSION     Minimum glibc label/check

Options:
  --build-dir DIR         Build directory, default: build/release-TIER-GLIBCMIN
  --dist-dir DIR          Output directory, default: dist
  --jobs N                Parallel build jobs, default: 4
  --cmake-extra ARGS      Extra CMake arguments, passed as one shell word
  --keep-build            Keep the existing build directory for partial rebuilds
  --allow-newer-glibc     Allow packaging when build-host glibc is newer than --glibc-min
  --help                  Show this help

The script packages non-glibc shared libraries next to the binary and writes a
bin/env-check helper that checks glibc and CPU features before execing punkst.
Run it on the oldest native Linux host you intend to support.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version=""
tier=""
glibc_min=""
build_dir=""
dist_dir="$repo_root/dist"
jobs=4
cmake_extra=()
allow_newer_glibc=0
keep_build=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --version)
      version="${2:-}"; shift 2 ;;
    --tier)
      tier="${2:-}"; shift 2 ;;
    --glibc-min)
      glibc_min="${2:-}"; shift 2 ;;
    --build-dir)
      build_dir="${2:-}"; shift 2 ;;
    --dist-dir)
      dist_dir="${2:-}"; shift 2 ;;
    --jobs)
      jobs="${2:-}"; shift 2 ;;
    --cmake-extra)
      cmake_extra+=("${2:-}"); shift 2 ;;
    --keep-build)
      keep_build=1; shift ;;
    --allow-newer-glibc)
      allow_newer_glibc=1; shift ;;
    --help)
      usage; exit 0 ;;
    *)
      printf 'Unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [ -z "$version" ] || [ -z "$tier" ]; then
  usage >&2
  exit 2
fi

case "$tier" in
  x86_64)
    cpu_flags=(-DENABLE_PORTABLE_BUILD=ON -DENABLE_NATIVE_ARCH=OFF -DENABLE_X86_64_V3=OFF -DENABLE_X86_64_V4=OFF)
    cpu_requirements="baseline x86_64"
    ;;
  x86_64-v3)
    cpu_flags=(-DENABLE_PORTABLE_BUILD=OFF -DENABLE_NATIVE_ARCH=OFF -DENABLE_X86_64_V3=ON -DENABLE_X86_64_V4=OFF)
    cpu_requirements="x86-64-v3: avx avx2 bmi1 bmi2 f16c fma lzcnt movbe"
    ;;
  x86_64-v4)
    cpu_flags=(-DENABLE_PORTABLE_BUILD=OFF -DENABLE_NATIVE_ARCH=OFF -DENABLE_X86_64_V3=OFF -DENABLE_X86_64_V4=ON)
    cpu_requirements="x86-64-v4: x86-64-v3 plus avx512f avx512bw avx512cd avx512dq avx512vl"
    ;;
  *)
    printf 'Unsupported tier: %s\n' "$tier" >&2
    exit 2 ;;
esac

version_gt() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    split(a, aa, "."); split(b, bb, ".");
    for (i = 1; i <= 3; i++) {
      av = aa[i] + 0; bv = bb[i] + 0;
      if (av > bv) exit 0;
      if (av < bv) exit 1;
    }
    exit 1;
  }'
}

host_glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}' || true)"
if [ -n "$host_glibc" ] && version_gt "$host_glibc" "$glibc_min" && [ "$allow_newer_glibc" -ne 1 ]; then
  printf 'ERROR: build host glibc is %s, but --glibc-min is %s.\n' "$host_glibc" "$glibc_min" >&2
  printf 'Build on a glibc %s host, raise --glibc-min, or pass --allow-newer-glibc for non-release smoke tests.\n' "$glibc_min" >&2
  exit 1
fi

if [ -z "$build_dir" ]; then
  build_dir="$repo_root/build/release-$tier-glibc$glibc_min"
fi

artifact_base="punkst-$version-linux-$tier-glibc$glibc_min"
stage_dir="$dist_dir/$artifact_base"
tarball="$dist_dir/$artifact_base.tar.gz"
manifest="$dist_dir/$version-linux-manifest.$artifact_base.jsonl"
checksum_file="$dist_dir/SHA256SUMS.$artifact_base"

printf 'Packaging %s\n' "$artifact_base"
printf 'Build directory: %s\n' "$build_dir"
printf 'Dist directory: %s\n' "$dist_dir"

cmake -S "$repo_root" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPUNKST_RUNTIME_OUTPUT_DIRECTORY="$build_dir/bin" \
  -DPUNKST_USE_ORIGIN_RPATH=ON \
  "${cpu_flags[@]}" \
  ${cmake_extra[@]+"${cmake_extra[@]}"}

cmake --build "$build_dir" --parallel "$jobs"

mkdir -p "$dist_dir"
if [ "$keep_build" -ne 1 ]; then
  rm -rf "$stage_dir"
fi
rm -f "$tarball"
mkdir -p "$stage_dir/bin" "$stage_dir/lib" "$stage_dir/docs"

cp "$build_dir/bin/punkst" "$stage_dir/bin/punkst"
cp "$repo_root/LICENSE" "$stage_dir/LICENSE"
cp "$repo_root/README.md" "$stage_dir/README.md"
cp -a "$repo_root/docs/." "$stage_dir/docs/"

cat > "$stage_dir/README.txt" <<EOF
punkst $version Linux prebuilt binary

CPU tier: $tier
Minimum glibc: $glibc_min
CPU requirements: $cpu_requirements

Recommended usage:
  ./bin/env-check --help

The real executable is ./bin/punkst. The env-check helper checks glibc and CPU
compatibility, then launches punkst with this tarball's bundled libraries.

Online documentation:
  https://yichen-si.github.io/punkst/
EOF

cat > "$stage_dir/BUILDINFO.txt" <<EOF
version=$version
tier=$tier
glibc_min=$glibc_min
cpu_requirements=$cpu_requirements
build_host=$(hostname 2>/dev/null || printf unknown)
build_date_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
compiler=$(${CXX:-c++} --version 2>/dev/null | head -n 1 || printf unknown)
cmake=$(cmake --version | head -n 1)
source_revision=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf unknown)
source_status=$(git -C "$repo_root" status --short 2>/dev/null | sed 's/"/\\"/g' || true)
EOF

ldd "$stage_dir/bin/punkst" |
  awk '
    /=> \// { print $(NF-1) }
    /^[[:space:]]*\// { print $1 }
  ' |
  while IFS= read -r lib; do
    base="$(basename "$lib")"
    case "$base" in
      linux-vdso.so.*|ld-linux*.so.*|libc.so.*|libm.so.*|libdl.so.*|libpthread.so.*|librt.so.*|libnsl.so.*|libresolv.so.*)
        continue ;;
    esac
    cp -L "$lib" "$stage_dir/lib/$base"
done

if command -v patchelf >/dev/null 2>&1; then
  patchelf --set-rpath '$ORIGIN/../lib' "$stage_dir/bin/punkst"
fi

cat > "$stage_dir/bin/env-check" <<EOF
#!/bin/sh
set -eu

tier="$tier"
glibc_min="$glibc_min"
root_dir=\$(CDPATH= cd -- "\$(dirname -- "\$0")/.." && pwd)

version_ge() {
  awk -v a="\$1" -v b="\$2" 'BEGIN {
    split(a, aa, "."); split(b, bb, ".");
    for (i = 1; i <= 3; i++) {
      av = aa[i] + 0; bv = bb[i] + 0;
      if (av > bv) exit 0;
      if (av < bv) exit 1;
    }
    exit 0;
  }'
}

glibc_version=\$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print \$2}' || true)
if [ -z "\$glibc_version" ]; then
  glibc_version=\$(ldd --version 2>/dev/null | sed -n '1s/.* //p' || true)
fi
if [ -z "\$glibc_version" ]; then
  printf 'env-check: could not determine glibc version. This tarball requires glibc >= %s.\\n' "\$glibc_min" >&2
  exit 1
fi
if ! version_ge "\$glibc_version" "\$glibc_min"; then
  printf 'env-check: glibc %s is too old; this tarball requires glibc >= %s.\\n' "\$glibc_version" "\$glibc_min" >&2
  exit 1
fi

flags=\$(sed -n 's/^flags[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -n 1 || true)
has_flag() {
  case " \$flags " in
    *" \$1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

need_flag() {
  if ! has_flag "\$1"; then
    printf 'env-check: CPU lacks required flag "%s" for %s. Use the baseline x86_64 tarball instead.\\n' "\$1" "\$tier" >&2
    exit 1
  fi
}

case "\$tier" in
  x86_64-v3)
    for f in avx avx2 bmi1 bmi2 f16c fma movbe; do need_flag "\$f"; done
    if ! has_flag lzcnt && ! has_flag abm; then
      printf 'env-check: CPU lacks required lzcnt/abm support for %s. Use the baseline x86_64 tarball instead.\\n' "\$tier" >&2
      exit 1
    fi ;;
  x86_64-v4)
    for f in avx avx2 bmi1 bmi2 f16c fma movbe avx512f avx512bw avx512cd avx512dq avx512vl; do need_flag "\$f"; done
    if ! has_flag lzcnt && ! has_flag abm; then
      printf 'env-check: CPU lacks required lzcnt/abm support for %s. Use the baseline x86_64 tarball instead.\\n' "\$tier" >&2
      exit 1
    fi ;;
esac

LD_LIBRARY_PATH="\$root_dir/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}" exec "\$root_dir/bin/punkst" "\$@"
EOF
chmod +x "$stage_dir/bin/env-check" "$stage_dir/bin/punkst"

runpath="$(objdump -p "$stage_dir/bin/punkst" | awk '/RUNPATH|RPATH/ {print $2}')"
case "$runpath" in
  *'$ORIGIN/../lib'*) ;;
  *)
    printf 'ERROR: punkst rpath/runpath does not contain $ORIGIN/../lib. Found: %s\n' "$runpath" >&2
    exit 1 ;;
esac
#case "$runpath" in
#  *"$HOME"*|*"/net/"*|*"/home/"*)
#    printf 'ERROR: punkst rpath/runpath contains a user-local path: %s\n' "$runpath" >&2
#    exit 1 ;;
#esac

"$stage_dir/bin/env-check" --help >/dev/null
ldd "$stage_dir/bin/punkst" > "$stage_dir/BUILDINFO.ldd.txt"
readelf --version-info "$stage_dir/bin/punkst" > "$stage_dir/BUILDINFO.readelf-version-info.txt"
objdump -p "$stage_dir/bin/punkst" > "$stage_dir/BUILDINFO.objdump-p.txt"

(
  cd "$dist_dir"
  tar -czf "$(basename "$tarball")" "$(basename "$stage_dir")"
  if [ -f "$checksum_file" ]; then
    grep -v "  $(basename "$tarball")\$" "$checksum_file" > "$checksum_file.tmp" || true
    mv "$checksum_file.tmp" "$checksum_file"
  fi
  sha256sum "$(basename "$tarball")" >> "$checksum_file"
)

cat >> "$manifest" <<EOF
{"version":"$version","tier":"$tier","glibc_min":"$glibc_min","asset":"$(basename "$tarball")","sha256_file":"SHA256SUMS"}
EOF

printf 'Wrote %s\n' "$tarball"
printf 'Updated %s\n' "$checksum_file"
printf 'Updated %s\n' "$manifest"
