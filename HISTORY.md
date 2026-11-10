# Repository history

In November 2026 this repository was reorganized into a `src/` layout, and
several separately developed repositories were merged into it **with their
full history**:

| Former repository | Now located at | Tag on its last commit before merging |
|---|---|---|
| priithon (this repository, GitHub) | `src/Priithon/` (was `Priithon/`) | `import/priithon-github` |
| Priithon (local development repository) | `src/Priithon/` | `import/priithon-local` |
| imgio | `src/Priithon/imgio/` | `import/imgio` |
| ndviewer | `src/Priithon/ndviewer/` | `import/ndviewer` |

The following modules were copied from the PriCommon and common repositories
**without history** (only the modules used by Chromagnon):

- PriCommon: `imgGeo`, `imgFilters`, `imgFit`, `xcorr`, `imgResample`, `flatConv`
- common: `commonfuncs`, `fntools`, `guiFuncs`, `listbox`, `microscopy`,
  `logger`, `ppro`, `ppro26`, `pprocess`

Added later for Optics (also without history):

- PriCommon: `region`
- common: `str2dic`

## Viewing the history of a file

Because the merged repositories kept their original paths in their own
history, `git log -- <current path>` only shows commits made after the merge.
To see the older commits, ask for the file at its **old path** starting from
the tag:

```bash
# imgio: src/Priithon/imgio/mrcIO.py
git log import/imgio -- mrcIO.py

# ndviewer: src/Priithon/ndviewer/main.py
git log import/ndviewer -- main.py

# Priithon (local development history): src/Priithon/useful.py
git log import/priithon-local -- useful.py

# Priithon (GitHub history before the merge): the package was in Priithon/
git log import/priithon-github -- Priithon/useful.py
```

To see the content of a file before the merge:

```bash
git show import/imgio:mrcIO.py
```

Files deleted during the reorganization (e.g. `imgio/Mrc_old.py`,
`imgio/tifffile_fix.rtf`) can be retrieved the same way:

```bash
git show import/imgio:Mrc_old.py > Mrc_old.py
```
