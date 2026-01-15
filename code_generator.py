#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLAUDE CODE CLONE (Gemini-Powered) - Enhanced with Code Indexing
A fully interactive coding agent with intelligent code indexing for minimal context.

New Features:
- 🗂️ Code Indexing: Builds symbol index of existing codebase
- 🎯 Smart Retrieval: Only sends relevant code to LLM
- 🔍 AST Analysis: Parses Python files to extract functions, classes, imports
- 📊 Dependency Graph: Tracks relationships between code entities
"""

import argparse
import os
import sys
import json
import subprocess
import re
import time
import ast
import hashlib
from typing import List, Tuple, Dict, Set, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import google.auth
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Try importing Vertex AI
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Content, Part, ChatSession
except ImportError:
    print("Error: google-cloud-aiplatform or rich not installed.")
    print("Run: pip install google-cloud-aiplatform rich")
    sys.exit(1)

console = Console()

# --- Code Indexing System ---

class CodeEntity:
    """Represents a code entity (function, class, method, etc.)"""
    def __init__(self, name: str, entity_type: str, file_path: str, 
                 line_start: int, line_end: int, signature: str = "",
                 docstring: str = "", parent: str = None):
        self.name = name
        self.entity_type = entity_type  # function, class, method, variable
        self.file_path = file_path
        self.line_start = line_start
        self.line_end = line_end
        self.signature = signature
        self.docstring = docstring
        self.parent = parent  # For methods, the parent class
        self.dependencies: Set[str] = set()  # What this entity depends on
        self.dependents: Set[str] = set()    # What depends on this entity
        
    def __repr__(self):
        return f"<{self.entity_type} {self.name} @ {self.file_path}:{self.line_start}>"

class PythonIndexer:
    """AST-based Python code indexer inspired by PySonar2"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.entities: Dict[str, CodeEntity] = {}  # key: qualified_name
        self.file_entities: Dict[str, List[str]] = defaultdict(list)  # file -> entity keys
        self.imports: Dict[str, Set[str]] = defaultdict(set)  # file -> imported modules
        self.file_hashes: Dict[str, str] = {}  # Track file changes
        
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash of file content for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def _qualified_name(self, name: str, parent: str = None, file_path: str = "") -> str:
        """Generate unique qualified name for entity"""
        if parent:
            return f"{parent}.{name}"
        # Use relative file path as namespace
        rel_path = str(Path(file_path).relative_to(self.base_dir)).replace('/', '.').replace('.py', '')
        return f"{rel_path}.{name}"
    
    def index_file(self, file_path: Path) -> bool:
        """Index a single Python file using AST"""
        try:
            # Check if file changed
            current_hash = self._get_file_hash(file_path)
            if str(file_path) in self.file_hashes and self.file_hashes[str(file_path)] == current_hash:
                return False  # No changes
            
            self.file_hashes[str(file_path)] = current_hash
            
            # Parse file
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source, filename=str(file_path))
            
            # Clear old entities for this file
            old_entities = self.file_entities.get(str(file_path), [])
            for entity_key in old_entities:
                self.entities.pop(entity_key, None)
            self.file_entities[str(file_path)] = []
            
            # Extract entities
            self._extract_entities(tree, str(file_path), source.split('\n'))
            return True
            
        except Exception as e:
            console.print(f"[dim red]Failed to index {file_path}: {e}[/dim red]")
            return False
    
    def _extract_entities(self, tree: ast.AST, file_path: str, lines: List[str], parent_class: str = None):
        """Recursively extract entities from AST"""
        for node in ast.walk(tree):
            try:
                # Extract functions
                if isinstance(node, ast.FunctionDef):
                    entity = self._create_function_entity(node, file_path, lines, parent_class)
                    if entity:
                        qualified_name = self._qualified_name(node.name, parent_class, file_path)
                        self.entities[qualified_name] = entity
                        self.file_entities[file_path].append(qualified_name)
                
                # Extract classes
                elif isinstance(node, ast.ClassDef):
                    entity = self._create_class_entity(node, file_path, lines)
                    if entity:
                        qualified_name = self._qualified_name(node.name, file_path=file_path)
                        self.entities[qualified_name] = entity
                        self.file_entities[file_path].append(qualified_name)
                        
                        # Extract methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method = self._create_function_entity(item, file_path, lines, qualified_name)
                                if method:
                                    method_key = self._qualified_name(item.name, qualified_name, file_path)
                                    self.entities[method_key] = method
                                    self.file_entities[file_path].append(method_key)
                
                # Extract imports
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._extract_imports(node, file_path)
                    
            except Exception as e:
                continue
    
    def _create_function_entity(self, node: ast.FunctionDef, file_path: str, 
                               lines: List[str], parent: str = None) -> Optional[CodeEntity]:
        """Create entity from function/method AST node"""
        try:
            # Get signature
            args = [arg.arg for arg in node.args.args]
            signature = f"{node.name}({', '.join(args)})"
            
            # Get docstring
            docstring = ast.get_docstring(node) or ""
            if docstring and len(docstring) > 200:
                docstring = docstring[:200] + "..."
            
            # Get line range
            line_start = node.lineno
            line_end = node.end_lineno or line_start
            
            entity_type = "method" if parent else "function"
            
            entity = CodeEntity(
                name=node.name,
                entity_type=entity_type,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                signature=signature,
                docstring=docstring,
                parent=parent
            )
            
            # Extract dependencies (called functions/classes)
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    entity.dependencies.add(n.func.id)
                elif isinstance(n, ast.Name):
                    entity.dependencies.add(n.id)
            
            return entity
        except:
            return None
    
    def _create_class_entity(self, node: ast.ClassDef, file_path: str, 
                            lines: List[str]) -> Optional[CodeEntity]:
        """Create entity from class AST node"""
        try:
            # Get base classes
            bases = [self._get_name(base) for base in node.bases]
            signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
            
            docstring = ast.get_docstring(node) or ""
            if docstring and len(docstring) > 200:
                docstring = docstring[:200] + "..."
            
            entity = CodeEntity(
                name=node.name,
                entity_type="class",
                file_path=file_path,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                signature=signature,
                docstring=docstring
            )
            
            # Track base class dependencies
            for base in bases:
                entity.dependencies.add(base)
            
            return entity
        except:
            return None
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""
    
    def _extract_imports(self, node: ast.AST, file_path: str):
        """Extract import information"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.imports[file_path].add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                self.imports[file_path].add(node.module)
    
    def build_dependency_graph(self):
        """Build reverse dependency graph (what depends on what)"""
        for entity_key, entity in self.entities.items():
            for dep in entity.dependencies:
                # Find matching entities
                for target_key, target in self.entities.items():
                    if target.name == dep:
                        target.dependents.add(entity_key)

class CodeIndexManager:
    """Manages code indexing and smart retrieval"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.indexer = PythonIndexer(base_dir)
        self.index_cache_file = self.base_dir / ".claude_code_index.json"
        
    def build_index(self, force: bool = False) -> int:
        """Build or update index of all Python files"""
        if not force and self._load_cache():
            console.print("[dim]Loaded index from cache[/dim]")
            return len(self.indexer.entities)
        
        indexed_count = 0
        for py_file in self.base_dir.rglob("*.py"):
            if '.venv' in str(py_file) or 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
            if self.indexer.index_file(py_file):
                indexed_count += 1
        
        self.indexer.build_dependency_graph()
        self._save_cache()
        
        return indexed_count
    
    def _save_cache(self):
        """Save index to cache file"""
        try:
            cache_data = {
                'entities': {
                    k: {
                        'name': v.name,
                        'type': v.entity_type,
                        'file': v.file_path,
                        'lines': [v.line_start, v.line_end],
                        'signature': v.signature,
                        'docstring': v.docstring,
                        'parent': v.parent
                    } for k, v in self.indexer.entities.items()
                },
                'file_hashes': self.indexer.file_hashes
            }
            with open(self.index_cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            console.print(f"[dim red]Failed to save cache: {e}[/dim red]")
    
    def _load_cache(self) -> bool:
        """Load index from cache file"""
        try:
            if not self.index_cache_file.exists():
                return False
            
            with open(self.index_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Verify files haven't changed
            for file_path, cached_hash in cache_data['file_hashes'].items():
                if not Path(file_path).exists():
                    return False
                current_hash = self.indexer._get_file_hash(Path(file_path))
                if current_hash != cached_hash:
                    return False  # Files changed, rebuild
            
            # Restore entities
            for key, data in cache_data['entities'].items():
                entity = CodeEntity(
                    name=data['name'],
                    entity_type=data['type'],
                    file_path=data['file'],
                    line_start=data['lines'][0],
                    line_end=data['lines'][1],
                    signature=data['signature'],
                    docstring=data['docstring'],
                    parent=data.get('parent')
                )
                self.indexer.entities[key] = entity
                self.indexer.file_entities[data['file']].append(key)
            
            self.indexer.file_hashes = cache_data['file_hashes']
            return True
        except Exception as e:
            console.print(f"[dim red]Failed to load cache: {e}[/dim red]")
            return False
    
    def search_entities(self, query: str, limit: int = 10) -> List[CodeEntity]:
        """Search for entities by name (fuzzy matching)"""
        query_lower = query.lower()
        matches = []
        
        for entity in self.indexer.entities.values():
            score = 0
            if query_lower in entity.name.lower():
                score = 100
            elif query_lower in entity.signature.lower():
                score = 50
            elif query_lower in entity.docstring.lower():
                score = 25
            
            if score > 0:
                matches.append((score, entity))
        
        matches.sort(reverse=True, key=lambda x: x[0])
        return [entity for _, entity in matches[:limit]]
    
    def get_relevant_code(self, query: str, max_tokens: int = 4000) -> str:
        """Retrieve relevant code snippets for a query"""
        # Search for relevant entities
        entities = self.search_entities(query, limit=20)
        
        if not entities:
            return ""
        
        # Build context with actual code
        context_parts = []
        current_tokens = 0
        seen_files = set()
        
        for entity in entities:
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            entity_info = self._format_entity_with_code(entity)
            entity_tokens = len(entity_info) // 4
            
            if current_tokens + entity_tokens > max_tokens:
                break
            
            context_parts.append(entity_info)
            current_tokens += entity_tokens
            seen_files.add(entity.file_path)
        
        if not context_parts:
            return ""
        
        header = f"## Relevant Code Context ({len(context_parts)} entities from {len(seen_files)} files)\n\n"
        return header + "\n\n".join(context_parts)
    
    def _format_entity_with_code(self, entity: CodeEntity) -> str:
        """Format entity with its actual code"""
        try:
            with open(entity.file_path, 'r') as f:
                lines = f.readlines()
            
            code = "".join(lines[entity.line_start-1:entity.line_end])
            
            header = f"### {entity.entity_type.upper()}: `{entity.signature}`\n"
            header += f"**File**: {Path(entity.file_path).relative_to(self.base_dir)}\n"
            if entity.docstring:
                header += f"**Doc**: {entity.docstring}\n"
            header += f"```python\n{code}```"
            
            return header
        except:
            return f"### {entity.entity_type.upper()}: {entity.name}\n(Code unavailable)"
    
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        stats = {
            'total_entities': len(self.indexer.entities),
            'files_indexed': len(self.indexer.file_entities),
            'by_type': defaultdict(int)
        }
        
        for entity in self.indexer.entities.values():
            stats['by_type'][entity.entity_type] += 1
        
        return stats

# --- Original Classes (Enhanced) ---

class CostTracker:
    """Tracks token usage and estimates cost for Gemini 2.0 Flash."""
    INPUT_PRICE_PER_M = 0.10
    OUTPUT_PRICE_PER_M = 0.40

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.start_time = time.time()
        self.context_savings = 0  # Tokens saved by indexing

    def update(self, usage_metadata):
        if usage_metadata:
            self.input_tokens += usage_metadata.prompt_token_count
            self.output_tokens += usage_metadata.candidates_token_count

    def add_savings(self, tokens: int):
        """Track context tokens saved by indexing"""
        self.context_savings += tokens

    def display(self):
        input_cost = (self.input_tokens / 1_000_000) * self.INPUT_PRICE_PER_M
        output_cost = (self.output_tokens / 1_000_000) * self.OUTPUT_PRICE_PER_M
        total_cost = input_cost + output_cost
        duration = time.time() - self.start_time

        table = Table(title="Session Usage & Cost (Est.)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Input Tokens", f"{self.input_tokens:,}")
        table.add_row("Output Tokens", f"{self.output_tokens:,}")
        table.add_row("Context Saved", f"{self.context_savings:,}", style="yellow")
        table.add_row("Total Cost", f"${total_cost:.4f}")
        table.add_row("Duration", f"{duration:.1f}s")
        
        console.print(table)

class AgentConfig:
    def __init__(self):
        self.model_name = 'gemini-2.0-flash-exp' 
        self.temperature = 0.1
        self.system_prompt = """
You are "Claude Code", an expert AI Software Engineer running in a CLI with code indexing.
You are interacting with a user to build, debug, or modify software.

CAPABILITIES:
1. 'list_files(path)': Explore file system.
2. 'read_file(path)': Read file contents.
3. 'write_file(path, content)': Create or overwrite files.
4. 'run_command(cmd)': Execute shell commands (git, grep, pytest, etc.).
5. CODE INDEX: When available, you'll receive relevant code context automatically.

RULES:
- RESEARCH first: Check the code index, list files, then read them. Don't guess.
- PLAN before acting: Briefly describe what you will do.
- FORMAT: Output tool calls in XML tags: <tool_code>function()</tool_code>.
- VERIFY: After editing, run tests or read the file back to confirm.
- CONTEXT: If a CLAUDE.md file exists, it contains project rules. Follow them.
- EFFICIENCY: The code index provides relevant context - use it to understand the codebase quickly.

EXAMPLE TOOL CALL:
<tool_code>
write_file("hello.py", '''
print("Hello World")
''')
</tool_code>
"""

class ToolSet:
    def __init__(self, base_dir: str, yolo_mode: bool = False):
        self.base_dir = os.path.abspath(base_dir)
        self.yolo_mode = yolo_mode

    def _is_safe_path(self, path: str) -> bool:
        full_path = os.path.abspath(os.path.join(self.base_dir, path))
        return full_path.startswith(self.base_dir)

    def _ask_permission(self, action: str, details: str) -> bool:
        if self.yolo_mode:
            return True
        console.print(f"\n[bold yellow]⚠️  PERMISSION REQUEST[/bold yellow]")
        console.print(f"Action: [bold]{action}[/bold]")
        console.print(f"Details: {details}")
        return Confirm.ask("Allow this action?")

    def list_files(self, path: str = ".") -> str:
        target = os.path.join(self.base_dir, path)
        if not os.path.exists(target): return f"Error: Path {path} not found."
        
        output = []
        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            level = root.replace(target, '').count(os.sep)
            if level > 2: continue
            indent = ' ' * 4 * level
            output.append(f"{indent}{os.path.basename(root)}/")
            for f in files:
                if not f.startswith('.'):
                    output.append(f"{' ' * 4 * (level + 1)}{f}")
        return "\n".join(output) if output else "(Empty directory)"

    def read_file(self, path: str) -> str:
        if not self._is_safe_path(path): return "Error: Access denied."
        try:
            with open(os.path.join(self.base_dir, path), 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e: return f"Error: {e}"

    def write_file(self, path: str, content: str) -> str:
        if not self._is_safe_path(path): return "Error: Access denied."
        
        full_path = os.path.join(self.base_dir, path)
        exists = os.path.exists(full_path)
        action_type = "UPDATE" if exists else "CREATE"
        
        snippet = content[:200] + "..." if len(content) > 200 else content
        if not self._ask_permission(f"{action_type} FILE", f"{path}\nContent Snippet:\n{snippet}"):
            return "Error: User denied permission."

        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e: return f"Error: {e}"

    def run_command(self, command: str) -> str:
        blocked = ['vim', 'nano', 'top', 'python3', 'python', 'node'] 
        cmd_start = command.strip().split()[0]
        if cmd_start in blocked and '-c' not in command and 'python' not in cmd_start:
             return "Error: Interactive commands (vim, nano) not supported."

        if not self._ask_permission("RUN COMMAND", command):
            return "Error: User denied permission."

        try:
            result = subprocess.run(
                command, shell=True, cwd=self.base_dir, 
                capture_output=True, text=True, timeout=120
            )
            out = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            return out[:5000] + "\n...(truncated)" if len(out) > 5000 else out
        except Exception as e: return f"Error: {e}"

class CodingAgent:
    def __init__(self, project_id: str, location: str, work_dir: str, yolo: bool = False, skip_index: bool = False):
        self.config = AgentConfig()
        self.tools = ToolSet(work_dir, yolo_mode=yolo)
        self.costs = CostTracker()
        self.work_dir = work_dir
        
        # Initialize code indexer
        if not skip_index:
            self.index_manager = CodeIndexManager(work_dir)
            console.print("[dim]Building code index...[/dim]")
            indexed = self.index_manager.build_index()
            stats = self.index_manager.get_statistics()
            console.print(f"[dim green]✓ Indexed {indexed} files, {stats['total_entities']} entities[/dim green]")
        else:
            self.index_manager = None
        
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(
            self.config.model_name,
            system_instruction=self.config.system_prompt
        )
        self.chat = self.model.start_chat()
        
        self._load_project_context()

    def _load_project_context(self):
        claude_md = os.path.join(self.work_dir, "CLAUDE.md")
        if os.path.exists(claude_md):
            try:
                with open(claude_md, 'r') as f:
                    content = f.read()
                self.chat.history.append(Content(role="user", parts=[Part.from_text(f"SYSTEM: Loaded CLAUDE.md project context:\n{content}")]))
                self.chat.history.append(Content(role="model", parts=[Part.from_text("Understood. I will follow the instructions in CLAUDE.md.")]))
                console.print(f"[dim]Loaded CLAUDE.md context[/dim]")
            except:
                pass

    def _extract_tool_calls(self, text: str) -> List[tuple]:
        matches = re.findall(r'<tool_code>(.*?)</tool_code>', text, re.DOTALL)
        calls = []
        for code in matches:
            try:
                code = code.strip()
                if code.startswith("list_files"):
                    calls.append(("list_files", re.search(r'\((.*?)\)', code).group(1).strip('"\' ')))
                elif code.startswith("read_file"):
                    calls.append(("read_file", re.search(r'\((.*?)\)', code).group(1).strip('"\' ')))
                elif code.startswith("run_command"):
                    calls.append(("run_command", re.search(r'\((.*?)\)', code).group(1).strip('"\' ')))
                elif code.startswith("write_file"):
                    first_paren = code.find('(')
                    last_paren = code.rfind(')')
                    inner = code[first_paren+1:last_paren]
                    parts = inner.split(',', 1)
                    path = parts[0].strip('"\' ')
                    content = parts[1].strip()
                    if content.startswith("'''") or content.startswith('"""'): content = content[3:-3]
                    if "\\n" in content: content = content.replace("\\n", "\n")
                    calls.append(("write_file", (path, content)))
            except: 
                pass
        return calls

    def step(self, user_input: str):
        # Check if we should add code context
        context_added = False
        original_input = user_input
        
        # Add relevant code context for feature requests
        if self.index_manager and not user_input.startswith("[SYSTEM]"):
            relevant_code = self.index_manager.get_relevant_code(user_input, max_tokens=4000)
            if relevant_code:
                # Estimate tokens saved
                try:
                    full_codebase_size = sum(
                        len(open(f, 'r').read()) for f in Path(self.work_dir).rglob("*.py")
                        if '.venv' not in str(f) and '__pycache__' not in str(f)
                    ) // 4  # Rough token estimate
                    context_size = len(relevant_code) // 4
                    self.costs.add_savings(max(0, full_codebase_size - context_size))
                except:
                    pass
                
                user_input = f"{user_input}\n\n{relevant_code}"
                context_added = True
                console.print(f"[dim]📎 Added relevant code context[/dim]")
        
        with Progress(SpinnerColumn(), TextColumn("[cyan]Thinking..."), transient=True) as progress:
            progress.add_task("request", total=None)
            try:
                response = self.chat.send_message(user_input)
                self.costs.update(response.usage_metadata)
                response_text = response.text
            except Exception as e:
                console.print(f"[red]API Error: {e}[/red]")
                return

        # Process Response
        tool_calls = self._extract_tool_calls(response_text)
        clean_text = re.sub(r'<tool_code>.*?</tool_code>', '', response_text, flags=re.DOTALL).strip()
        
        if clean_text:
            console.print(Panel(Markdown(clean_text), title="Claude Code", border_style="blue"))

        # Execute Tools
        if tool_calls:
            for tool_name, args in tool_calls:
                console.print(f"[dim]➜ Executing {tool_name}...[/dim]")
                result = ""
                if tool_name == "list_files": 
                    result = self.tools.list_files(args or ".")
                elif tool_name == "read_file": 
                    result = self.tools.read_file(args)
                elif tool_name == "write_file": 
                    path, content = args
                    result = self.tools.write_file(path, content)
                    # Re-index if Python file was modified
                    if self.index_manager and path.endswith('.py'):
                        self.index_manager.indexer.index_file(Path(self.work_dir) / path)
                        console.print("[dim]Updated code index[/dim]")
                elif tool_name == "run_command": 
                    result = self.tools.run_command(args)
                
                self.step(f"[SYSTEM] TOOL_OUTPUT for {tool_name}:\n{result}")
    
    def show_index_stats(self):
        """Display index statistics"""
        if not self.index_manager:
            console.print("[yellow]Index not available[/yellow]")
            return
        
        stats = self.index_manager.get_statistics()
        
        table = Table(title="Code Index Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green")
        
        table.add_row("Total Entities", str(stats['total_entities']))
        table.add_row("Files Indexed", str(stats['files_indexed']))
        
        for entity_type, count in stats['by_type'].items():
            table.add_row(f"  {entity_type.title()}s", str(count))
        
        console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Claude Code Clone with Code Indexing")
    parser.add_argument('--project-dir', default=os.getcwd(), help="Project directory to work in")
    parser.add_argument('--service-account', required=False, help="Path to service account JSON")
    parser.add_argument('--project-id', required=False, help="GCP project ID")
    parser.add_argument('--yolo', action='store_true', help="Skip permission prompts")
    parser.add_argument('--rebuild-index', action='store_true', help="Force rebuild code index")
    parser.add_argument('--skip-index', action='store_true', help="Skip code indexing")
    args = parser.parse_args()

    # Auth
    project_id = args.project_id
    if args.service_account:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.service_account
        with open(args.service_account) as f: 
            project_id = json.load(f).get('project_id')
    if not project_id:
        try: 
            _, project_id = google.auth.default()
        except: 
            console.print("[red]Error: Project ID not found.[/red]")
            console.print("Provide --project-id or set GOOGLE_APPLICATION_CREDENTIALS")
            return

    # Header
    console.rule("[bold blue]Claude Code (Gemini Ed.) with Code Indexing[/bold blue]")
    console.print(f"Dir: [bold]{args.project_dir}[/bold] | YOLO Mode: {args.yolo}")
    console.print("Commands: /cost, /clear, /help, /index, exit")
    
    agent = CodingAgent(project_id, 'us-central1', args.project_dir, 
                       yolo=args.yolo, skip_index=args.skip_index)
    
    # Force rebuild if requested
    if args.rebuild_index and agent.index_manager:
        console.print("[yellow]Rebuilding index...[/yellow]")
        agent.index_manager.build_index(force=True)
        console.print("[green]✓ Index rebuilt[/green]")
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]>[/bold green]")
            if not user_input.strip(): 
                continue

            # Slash Commands
            if user_input.lower() in ['exit', 'quit']: 
                break
            
            if user_input.lower() == '/cost': 
                agent.costs.display()
                continue
            
            if user_input.lower() == '/help':
                console.print("""
Available commands:
  /cost    - Show token usage and cost
  /index   - Show code index statistics
  /clear   - Reset conversation memory
  /compact - Summarize history (not implemented)
  exit     - Quit the program
""")
                continue
            
            if user_input.lower() == '/index':
                agent.show_index_stats()
                continue
            
            if user_input.lower() == '/clear':
                agent = CodingAgent(project_id, 'us-central1', args.project_dir, 
                                  yolo=args.yolo, skip_index=args.skip_index)
                console.print("[yellow]Memory cleared.[/yellow]")
                continue

            agent.step(user_input)
            
        except KeyboardInterrupt: 
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

    console.print("\n[blue]Goodbye![/blue]")

if __name__ == '__main__':
    main()
