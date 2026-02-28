# Windows Compatibility Fixes

## Changes

### Fix DocsManual.tsx generation on Windows (CRLF line endings)

The `manual.Run()` doc generator reads `.manual/*.md` chapter files and generates `DocsManual.tsx`. On Windows, these files have `\r\n` line endings. `strings.Split(content, "\n")` preserves trailing `\r` on each line, which leaks into generated anchor IDs and JSX string literals (e.g. `href="#chapter-1-introduction\r"`), producing ~1157 TypeScript compilation errors.

**Fix:** Normalize `\r\n` to `\n` after reading each chapter file in `loadManualContent()`.

**File:** `cmd/server/api/tooling/docs/manual/manual.go`
