#!/usr/bin/env bash
# Быстрый поиск по исходникам репозитория (рипгрэп поверх grep).
#
# - обходит build-каталоги (следует .gitignore: /build*, *.o, *.so),
# - не заходит в .git/, модели и мусор,
# - читает только текст (не виснет на бинарных артефактах).
#
# Примеры:
#   scripts/srcgrep.sh "mul_mat_vec_q"
#   scripts/srcgrep.sh -n "GGML_TYPE_MXFP4" ggml/src
#   scripts/srcgrep.sh -i "prefetch" ggml/src/ggml-cuda/

set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$ROOT" ]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

# --hidden: ищем и в скрытых файлах (если явно указано), но исключаем git/build.
# .gitignore у нас уже покрывает build* / *.o / *.so — rg их не читает.
exec rg --hidden --glob '!.git/**' "$@"
