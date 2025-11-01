"""Parser entry point for the Factorio Circuit DSL."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from lark import Lark
from lark.exceptions import LexError, ParseError

from dsl_compiler.src.ast import ASTNode, Program
from .preprocessor import preprocess_imports
from .transformer import DSLTransformer


class DSLParser:
    """Main parser class for the Factorio Circuit DSL."""

    def __init__(self, grammar_path: Optional[Path] = None):
        """Initialize parser with grammar file."""
        if grammar_path is None:
            grammar_path = (
                Path(__file__).resolve().parent.parent.parent / "grammar" / "fcdsl.lark"
            )

        self.grammar_path = grammar_path
        self.parser = None
        self.transformer = DSLTransformer()
        self._load_grammar()

    def _load_grammar(self) -> None:
        """Load and compile the Lark grammar."""
        try:
            with open(self.grammar_path, "r") as handle:
                grammar_text = handle.read()

            self.parser = Lark(
                grammar_text,
                parser="lalr",
                transformer=self.transformer,
                start="start",
                debug=False,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Grammar file not found: {self.grammar_path}"
            ) from exc
        except Exception as exc:  # pragma: no cover - unexpected
            raise RuntimeError(f"Failed to load grammar: {exc}") from exc

    def parse(self, source_code: str, filename: str = "<string>") -> Program:
        """Parse DSL source code into an AST.

        Args:
            source_code: The source code text to parse
            filename: Source file path for error reporting and import resolution

        Returns:
            Program AST node representing the parsed source

        Raises:
            SyntaxError: If the source code has parse errors
            RuntimeError: If parser is not initialized or unexpected error occurs
        """
        if self.parser is None:
            raise RuntimeError("Parser not initialized")

        try:
            if filename != "<string>":
                file_path = Path(filename)
                if not file_path.is_absolute():
                    file_path = (Path.cwd() / file_path).resolve()
                base_path = file_path.parent
            else:
                base_path = Path("tests/sample_programs").resolve()

            preprocessed_code = preprocess_imports(source_code, base_path)
            ast = self.parser.parse(preprocessed_code)
            self._attach_source_file(ast, filename)

            if not isinstance(ast, Program):
                raise RuntimeError(f"Expected Program AST node, got {type(ast)}")

            return ast

        except (ParseError, LexError) as exc:
            raise SyntaxError(f"Parse error in {filename}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Unexpected error parsing {filename}: {exc}") from exc

    def parse_file(self, file_path: Path) -> Program:
        """Parse a DSL file into an AST."""
        try:
            with open(file_path, "r") as handle:
                source_code = handle.read()
            return self.parse(source_code, str(file_path))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Source file not found: {file_path}") from exc

    def _attach_source_file(self, node: ASTNode, filename: str) -> None:
        """Recursively annotate AST nodes with their originating filename."""
        if not isinstance(node, ASTNode):
            return

        if filename:
            node.source_file = filename

        for attr in vars(node).values():
            if isinstance(attr, ASTNode):
                self._attach_source_file(attr, filename)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, ASTNode):
                        self._attach_source_file(item, filename)
