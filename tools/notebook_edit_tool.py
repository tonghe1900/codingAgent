"""
tools/notebook_edit_tool.py
============================
NotebookEditTool — reads and edits Jupyter Notebook (.ipynb) files.

Jupyter notebooks are JSON files.  This tool exposes three operations:
* ``read``       — return all cells with their source and outputs.
* ``edit_cell``  — replace the source of a specific cell (by index).
* ``insert_cell``— insert a new cell at a given index.
* ``delete_cell``— remove a cell by index.

Cell indices are 0-based and validated before any write is made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from claude_code.tool import Tool, ToolResult, PermissionDecision

CELL_TYPES = {"code", "markdown", "raw"}


class NotebookEditTool(Tool):
    """Read and edit Jupyter Notebook (.ipynb) files."""

    name = "NotebookEdit"
    description = (
        "Read or edit a Jupyter Notebook (.ipynb) file. "
        "Supports read (view cells), edit_cell (replace source), "
        "insert_cell (add a new cell), and delete_cell operations."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "notebook_path": {
                "type": "string",
                "description": "Absolute path to the .ipynb file.",
            },
            "operation": {
                "type": "string",
                "enum": ["read", "edit_cell", "insert_cell", "delete_cell"],
                "description": "Operation to perform.",
            },
            "cell_index": {
                "type": "integer",
                "description": "0-based cell index (required for edit/insert/delete).",
            },
            "new_source": {
                "type": "string",
                "description": "Replacement source code/text (required for edit_cell and insert_cell).",
            },
            "cell_type": {
                "type": "string",
                "enum": ["code", "markdown", "raw"],
                "description": "Cell type for insert_cell (default 'code').",
            },
        },
        "required": ["notebook_path", "operation"],
    }
    dangerous = True   # Modifies files

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        decision = self.check_permission(input_data, permission_manager)
        if decision == PermissionDecision.DENY:
            return ToolResult.error("Permission denied: notebook edit was blocked.")

        nb_path = Path(input_data.get("notebook_path", "")).expanduser()
        op      = input_data.get("operation", "read")

        if not nb_path.name.endswith(".ipynb"):
            return ToolResult.error("Only .ipynb files are supported.")

        if op == "read":
            return self._read(nb_path)
        if op == "edit_cell":
            return self._edit_cell(nb_path, input_data)
        if op == "insert_cell":
            return self._insert_cell(nb_path, input_data)
        if op == "delete_cell":
            return self._delete_cell(nb_path, input_data)

        return ToolResult.error(f"Unknown operation '{op}'.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(nb_path: Path) -> tuple[Optional[Dict], Optional[ToolResult]]:
        """Load notebook JSON, returning (data, None) or (None, error_result)."""
        if not nb_path.exists():
            return None, ToolResult.error(f"Notebook not found: {nb_path}")
        try:
            return json.loads(nb_path.read_text(encoding="utf-8")), None
        except (json.JSONDecodeError, OSError) as exc:
            return None, ToolResult.error(f"Could not read notebook: {exc}")

    @staticmethod
    def _save(nb_path: Path, data: Dict) -> Optional[ToolResult]:
        """Save notebook JSON, returning an error ToolResult on failure."""
        try:
            nb_path.write_text(json.dumps(data, indent=1), encoding="utf-8")
            return None
        except OSError as exc:
            return ToolResult.error(f"Could not write notebook: {exc}")

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _read(self, nb_path: Path) -> ToolResult:
        data, err = self._load(nb_path)
        if err:
            return err

        cells: List[str] = []
        for i, cell in enumerate(data.get("cells", [])):
            ctype  = cell.get("cell_type", "code")
            source = "".join(cell.get("source", []))
            output_lines: List[str] = []
            for out in cell.get("outputs", []):
                otype = out.get("output_type", "")
                if otype in ("stream", "display_data", "execute_result"):
                    text = out.get("text", []) or out.get("data", {}).get("text/plain", [])
                    output_lines.append("".join(text))
            output_summary = ("\n  Output:\n    " + "\n    ".join(output_lines)) if output_lines else ""
            cells.append(f"[{i}] {ctype}\n{source}{output_summary}")

        return ToolResult.ok(
            f"Notebook: {nb_path}\n\n" + "\n\n---\n\n".join(cells),
            cell_count=len(data.get("cells", [])),
        )

    def _edit_cell(self, nb_path: Path, data_in: Dict) -> ToolResult:
        idx = data_in.get("cell_index")
        new_source = data_in.get("new_source")
        if idx is None:
            return ToolResult.error("'cell_index' is required.")
        if new_source is None:
            return ToolResult.error("'new_source' is required.")

        data, err = self._load(nb_path)
        if err:
            return err
        cells = data.get("cells", [])
        if not (0 <= idx < len(cells)):
            return ToolResult.error(f"cell_index {idx} is out of range (0–{len(cells)-1}).")

        cells[idx]["source"] = new_source.splitlines(keepends=True)
        cells[idx]["outputs"] = []   # clear outputs like Jupyter does on edit
        save_err = self._save(nb_path, data)
        return save_err or ToolResult.ok(f"Edited cell {idx} in {nb_path}.")

    def _insert_cell(self, nb_path: Path, data_in: Dict) -> ToolResult:
        idx        = data_in.get("cell_index", 0)
        new_source = data_in.get("new_source", "")
        cell_type  = data_in.get("cell_type", "code")
        if cell_type not in CELL_TYPES:
            return ToolResult.error(f"Invalid cell_type '{cell_type}'.")

        data, err = self._load(nb_path)
        if err:
            return err
        cells = data.get("cells", [])
        new_cell: Dict[str, Any] = {
            "cell_type": cell_type,
            "metadata":  {},
            "source":    new_source.splitlines(keepends=True),
            "outputs":   [] if cell_type == "code" else None,
            "execution_count": None,
        }
        if cell_type != "code":
            new_cell.pop("outputs", None)
            new_cell.pop("execution_count", None)

        cells.insert(idx, new_cell)
        save_err = self._save(nb_path, data)
        return save_err or ToolResult.ok(f"Inserted {cell_type} cell at index {idx}.")

    def _delete_cell(self, nb_path: Path, data_in: Dict) -> ToolResult:
        idx = data_in.get("cell_index")
        if idx is None:
            return ToolResult.error("'cell_index' is required.")

        data, err = self._load(nb_path)
        if err:
            return err
        cells = data.get("cells", [])
        if not (0 <= idx < len(cells)):
            return ToolResult.error(f"cell_index {idx} is out of range (0–{len(cells)-1}).")

        cells.pop(idx)
        save_err = self._save(nb_path, data)
        return save_err or ToolResult.ok(f"Deleted cell {idx} from {nb_path}.")
