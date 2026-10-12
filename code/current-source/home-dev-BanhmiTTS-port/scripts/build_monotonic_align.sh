#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
package_dir="${project_root}/src/python/banhmi_tts/model/monotonic_align"
cythonize_bin="${CYTHONIZE:-cythonize}"

cd "${package_dir}"
"${cythonize_bin}" -i _core.pyx
