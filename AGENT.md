# General Agent Guidelines

# running bash programs

When running programs, be aware that there is a venv in the factompiler directory that needs to be activated first to access the required dependencies. You can activate it by running once:

```bash
source /path/to/factompiler/venv/bin/activate
```

# draftsman

The compiler emits draftsman code for the designs. Draftsman is a Python library for creating DXF files. You can find more information about Draftsman and its usage in the [Draftsman documentation](https://draftsman.readthedocs.io/en/latest/).

Its source code is available in the directory ../draftsman relative to the factompiler directory.
But be aware that this is just for reference, and you should use the installed version in the venv.

# sample programs

There is a collection of sample programs available in the `tests/sample_programs` directory within the factompiler repository. These programs denote what we want to be able to compile. You can refer to these samples for guidance on how to structure your own programs. They should only be changed if we remove or add new features to the language.


# language spec

The `Language_SPEC.md` file in the factompiler repository contains the complete specification of the Factom programming language. This document outlines the syntax, semantics, and features of the language. It serves as a reference for understanding how to write programs in Factom and how the compiler should interpret them.