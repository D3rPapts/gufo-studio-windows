# Bundled Microsoft runtime DLLs

`gufo.exe` imports these at load time, and the Windows package ships them
app-local so the archive runs on a machine without Visual Studio or the
Visual C++ redistributable. Missing `libomp140.x86_64.dll` in particular
produced a silent `0xC0000135` exit before this directory existed.

| File | Version | Source |
|---|---|---|
| `vcruntime140.dll` | 14.51.36247.0 | VC Redist `14.51.36231/x64/Microsoft.VC145.CRT` |
| `vcruntime140_1.dll` | 14.51.36247.0 | VC Redist `14.51.36231/x64/Microsoft.VC145.CRT` |
| `msvcp140.dll` | 14.51.36247.0 | VC Redist `14.51.36231/x64/Microsoft.VC145.CRT` |
| `msvcp140_atomic_wait.dll` | 14.51.36247.0 | VC Redist `14.51.36231/x64/Microsoft.VC145.CRT` |
| `libomp140.x86_64.dll` | LLVM OpenMP (FileVersion 20140926) | Visual Studio `Microsoft.VC145.OpenMP.LLVM` runtime, byte-identical to the copy the VS installer lays down in `System32` |

The four CRT DLLs are Microsoft Visual C++ Redistributable code and are
redistributable with applications under the Microsoft Software License
Terms. `libomp140.x86_64.dll` is Microsoft's build of the LLVM OpenMP
runtime, whose upstream source is Apache-2.0 WITH LLVM-exception.

Refresh these files when the Visual Studio version used for release builds
changes, keeping the CRT DLLs at or above the toolchain that linked
`gufo.exe`.
