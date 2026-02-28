# Windows Compatibility Fixes

## Changes

### Fix DocsManual.tsx generation on Windows (CRLF line endings)

The `manual.Run()` doc generator reads `.manual/*.md` chapter files and generates `DocsManual.tsx`. On Windows, these files have `\r\n` line endings. `strings.Split(content, "\n")` preserves trailing `\r` on each line, which leaks into generated anchor IDs and JSX string literals (e.g. `href="#chapter-1-introduction\r"`), producing ~1157 TypeScript compilation errors.

**Fix:** Normalize `\r\n` to `\n` after reading each chapter file in `loadManualContent()`.

**File:** `cmd/server/api/tooling/docs/manual/manual.go`

### Fix makefile SHELL detection and POSIX compatibility on Windows

Two issues prevented the makefile from working on Windows (Git Bash / MSYS2):

1. **SHELL detection:** The `$(shell which bash ...)` call fails under `mingw32-make` because it evaluates `$(shell)` using its default `sh.exe`, where `which` isn't reliably available. This caused all recipes to run under `sh.exe` instead of `bash.exe`, breaking pipe handling. Fix: detect `Windows_NT` via `$(OS)` and derive `bash.exe` from the default `sh.exe` path using `$(subst)`.

2. **`source` → `.`:** The `source .env` command is a Bash builtin not supported by POSIX `sh`. Replaced with `.` (dot), the POSIX-portable equivalent that works in `sh`, `bash`, and `ash`.

**File:** `makefile`
