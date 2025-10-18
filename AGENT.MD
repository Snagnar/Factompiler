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

# coding guidelines

THIS PACKAGE IS NOT RELEASED YET. Ergo, you may never ever need to keep backward compability in mind and in the code. If I tell you to change something, change it and do not regard anything that was there before.

When writing code, please follow these guidelines:
- Write clear and concise code with appropriate comments.
- Follow the existing coding style and conventions used in the project.
- Write unit tests for new features or bug fixes to ensure code quality.
- Do not use defensive programming unless explicitly told to do so.
- Letting things fail/throw exceptions is often better than trying to handle every possible error case.