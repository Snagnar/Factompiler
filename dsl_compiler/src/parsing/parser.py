from __future__ import annotations

from pathlib import Path

from lark import Lark
from lark.exceptions import LexError, ParseError, UnexpectedEOF, UnexpectedInput

from dsl_compiler.src.ast.statements import ASTNode, Program

from .preprocessor import LineMap, preprocess_imports_with_map
from .transformer import DSLTransformer

"""Parser entry point for the Facto language."""


# Human-readable names for grammar terminals used in "expected ..." hints.
_TOKEN_NAMES = {
    "SEMICOLON": "';'",
    "COMMA": "','",
    "COLON": "':'",
    "DOT": "'.'",
    "EQUAL": "'='",
    "LPAR": "'('",
    "RPAR": "')'",
    "LBRACE": "'{'",
    "RBRACE": "'}'",
    "LSQB": "'['",
    "RSQB": "']'",
    "NAME": "a name",
    "NUMBER": "a number",
    "STRING": "a string",
    "UNARY_OP": "'-' / '!'",
    "ADD_OP": "'+' / '-'",
    "MUL_OP": "'*' / '/' / '%'",
    "COMP_OP": "a comparison",
    "OUTPUT_SPEC": "':'",
    "PROJ_OP": "'|'",
    "RANGE_OP": "'..'",
    "WHEN_KW": "'when'",
    "SET_KW": "'set'",
    "RESET_KW": "'reset'",
    "$END": "end of input",
}


class DSLParser:
    """Main parser class for the Facto language."""

    def __init__(self, grammar_path: Path | None = None):
        """Initialize parser with grammar file."""
        if grammar_path is None:
            grammar_path = Path(__file__).resolve().parent.parent.parent / "grammar" / "facto.lark"

        self.grammar_path = grammar_path
        self.parser: Lark | None = None
        self.transformer = DSLTransformer()
        self._load_grammar()

    def _load_grammar(self) -> None:
        """Load and compile the Lark grammar."""
        try:
            with open(self.grammar_path) as handle:
                grammar_text = handle.read()

            self.parser = Lark(
                grammar_text,
                parser="lalr",
                transformer=self.transformer,
                start="start",
                debug=False,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Grammar file not found: {self.grammar_path}") from exc
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
            RuntimeError: If unexpected error occurs
        """
        try:
            if filename != "<string>":
                file_path = Path(filename)
                if not file_path.is_absolute():
                    file_path = (Path.cwd() / file_path).resolve()
                base_path = file_path.parent
            else:
                base_path = Path("example_programs").resolve()

            preprocessed_code, line_map = preprocess_imports_with_map(
                source_code, base_path, source_name=filename
            )
            if self.parser is None:
                raise RuntimeError("Parser not initialized")
            try:
                ast = self.parser.parse(preprocessed_code)
            except UnexpectedInput as exc:
                raise SyntaxError(
                    self._format_syntax_error(exc, preprocessed_code, line_map, filename)
                ) from exc

            if not isinstance(ast, Program):
                raise RuntimeError(f"Expected Program AST node, got {type(ast)}")

            self._attach_source_file(ast, filename, line_map)

            return ast

        except SyntaxError:
            raise
        except FileNotFoundError as exc:
            # Missing import: a user error, reported like a syntax error
            raise SyntaxError(f"Import error in {exc}") from exc
        except (ParseError, LexError) as exc:
            raise SyntaxError(f"Parse error in {filename}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Unexpected error parsing {filename}: {exc}") from exc

    def parse_file(self, file_path: Path) -> Program:
        """Parse a DSL file into an AST."""
        try:
            with open(file_path) as handle:
                source_code = handle.read()
            return self.parse(source_code, str(file_path))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Source file not found: {file_path}") from exc

    @staticmethod
    def _format_syntax_error(
        exc: UnexpectedInput, code: str, line_map: LineMap, filename: str
    ) -> str:
        """Turn a Lark parse error into a short, readable message.

        Example::

            Syntax error in main.facto:2:1: unexpected 'Signal'
              Signal b = 3;
              ^
            Did you forget a ';' at the end of the previous line?
        """
        line = getattr(exc, "line", None) or 0
        column = getattr(exc, "column", None) or 0
        src_file, src_line = filename, line
        if 0 < line <= len(line_map):
            mapped_file, src_line = line_map[line - 1]
            src_file = mapped_file or filename
        where = Path(src_file).name if src_file and src_file != "<string>" else "<input>"
        if src_line:
            where += f":{src_line}"
            if column:
                where += f":{column}"

        if isinstance(exc, UnexpectedEOF):
            what = "unexpected end of input"
        elif getattr(exc, "token", None) is not None:
            tok = exc.token  # type: ignore[attr-defined]
            what = f"unexpected '{tok}'" if str(tok) else "unexpected end of input"
        elif getattr(exc, "char", None) is not None:
            what = f"unexpected character '{exc.char}'"  # type: ignore[attr-defined]
        else:
            what = "invalid syntax"

        message = f"Syntax error in {where}: {what}"

        lines = code.split("\n")
        if 0 < line <= len(lines):
            text = lines[line - 1]
            message += f"\n  {text.rstrip()}\n  {' ' * max(column - 1, 0)}^"

        expected = set(getattr(exc, "expected", None) or getattr(exc, "allowed", None) or [])
        prev_tokens = [str(t) for t in (getattr(exc, "token_history", None) or [])]
        if "SEMICOLON" in expected and prev_tokens and prev_tokens[-1] != ";":
            message += "\nDid you forget a ';' at the end of the previous statement?"
        elif expected and len(expected) <= 8:
            names = sorted({_TOKEN_NAMES.get(t, t) for t in expected})
            message += "\nExpected one of: " + ", ".join(names)
        return message

    def _attach_source_file(
        self, node: ASTNode, filename: str, line_map: LineMap | None = None
    ) -> None:
        """Annotate AST nodes with their originating file and original line.

        Imports are inlined before parsing, so Lark's line numbers refer to the
        expanded code. ``line_map`` translates them back to (file, line).
        """
        seen: set[int] = set()

        def visit(n: ASTNode) -> None:
            if id(n) in seen:
                return
            seen.add(id(n))
            line = getattr(n, "line", 0) or 0
            if line_map and 0 < line <= len(line_map):
                mapped_file, mapped_line = line_map[line - 1]
                n.line = mapped_line
                n.source_file = mapped_file or filename
            elif filename:
                n.source_file = filename

            for attr in vars(n).values():
                if isinstance(attr, ASTNode):
                    visit(attr)
                elif isinstance(attr, list):
                    for item in attr:
                        if isinstance(item, ASTNode):
                            visit(item)

        if isinstance(node, ASTNode):
            visit(node)
