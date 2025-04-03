import ast
from langchain.schema import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Split text into chunks

def chunk_text_and_add_metadata(texts, references, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []

    for text, reference in zip(texts, references):
        chunks.extend([
            Document(
                page_content=chunk,
                metadata={
                    "source": reference,
                }
            ) 
            for chunk in text_splitter.split_text(text)
        ])
    return chunks


def chunk_pythoncode_and_add_metadata(code_files_content, code_files_path, max_chunk_size):
    chunks = [] 
    for code_file_content, code_file_path in zip(code_files_content, code_files_path):
        document_chunks = _generate_code_chunks_with_metadata(code_file_content, code_file_path, max_chunk_size)
        chunks.extend(document_chunks)   
    return chunks

def _generate_code_chunks_with_metadata(code_file_content, code_file_path, max_chunk_size):
    documents = []

    _iterate_ast(code_file_content, documents, code_file_path, max_chunk_size)
    usage = None
    if code_file_path.startswith("kadi_apy/lib/"):
        usage = "kadi_apy/lib/"
    elif code_file_path.startswith("kadi_apy/cli/"):
        usage = "kadi_apy/cli/"
      

    for doc in documents:
        doc.metadata["source"] = code_file_path
        if usage is not None:
            doc.metadata["usage"] = usage

    return documents

def _iterate_ast(code_file_content, documents, code_file_path, max_chunk_size):
    tree = ast.parse(code_file_content, filename=code_file_path)
    first_level_nodes = list(ast.iter_child_nodes(tree))

    if not first_level_nodes:
        documents.extend(
            _chunk_nodeless_code_file_content(code_file_content, code_file_path, max_chunk_size))
        return

    all_imports = all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in first_level_nodes)

    if all_imports:
        documents.extend(
            _chunk_import_only_code_file_content(code_file_content, code_file_path, max_chunk_size)
        )
    else:
        for first_level_node in ast.iter_child_nodes(tree):
            if isinstance(first_level_node, ast.ClassDef):
                documents.extend(
                    _handle_first_level_class(first_level_node, code_file_content, max_chunk_size)
                )
            elif isinstance(first_level_node, ast.FunctionDef):
                documents.extend(
                    _chunk_first_level_func_node(first_level_node, code_file_content, max_chunk_size)
                )
            elif isinstance(first_level_node, ast.Assign):
                documents.extend(
                    _chunk_first_level_assign_node(first_level_node, code_file_content, max_chunk_size)
                )
            else:
                if not isinstance(first_level_node, (ast.Import, ast.ImportFrom)):
                    documents.extend(
                        _handle_not_defined_case(first_level_node, code_file_content, max_chunk_size)
                    )

def _handle_first_level_class(ast_node, code_file_content, max_chunk_size):
    documents = []
    class_start_line = ast_node.lineno
    class_body_lines = [child.lineno for child in ast_node.body if isinstance(child, ast.FunctionDef)]
    class_end_line = min(class_body_lines, default=ast_node.end_lineno) - 1
    class_source = '\n'.join(code_file_content.splitlines()[class_start_line-1:class_end_line])

    metadata = {
        "type": "class",
        "class": ast_node.name,
        "visibility": "public"
    }

    if len(class_source) > max_chunk_size:
        class_chunks = _split_into_chunks(class_source, max_chunk_size)
        for chunk in class_chunks:
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
    else:
        doc = Document(
            page_content=class_source,
            metadata=metadata
        )
        documents.append(doc)

    for second_level_node in ast.iter_child_nodes(ast_node):
        if isinstance(second_level_node, ast.FunctionDef):
            method_start_line = (
                second_level_node.decorator_list[0].lineno
                if second_level_node.decorator_list else second_level_node.lineno
            )
            method_end_line = second_level_node.end_lineno
            method_source = '\n'.join(code_file_content.splitlines()[method_start_line-1:method_end_line])

            visibility = "internal" if second_level_node.name.startswith("_") else "public"

            if len(method_source) > max_chunk_size:
                method_chunks = _split_into_chunks(method_source, max_chunk_size)
                for chunk in method_chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "type": "method",
                            "method": second_level_node.name,
                            "visibility": visibility,
                            "class": ast_node.name
                        }
                    )
                    documents.append(doc)
            else:
                doc = Document(
                    page_content=method_source,
                    metadata={
                        "type": "method",
                        "method": second_level_node.name,
                        "visibility": visibility,
                        "class": ast_node.name
                    }
                )
                documents.append(doc)

    return documents

def _chunk_first_level_func_node(ast_node, code_file_content, max_chunk_size):
    documents = []
    function_start_line = (
        ast_node.decorator_list[0].lineno
        if ast_node.decorator_list else ast_node.lineno
    )
    function_end_line = ast_node.end_lineno
    function_source = '\n'.join(code_file_content.splitlines()[function_start_line-1:function_end_line])

    visibility = "internal" if ast_node.name.startswith("_") else "public"

    is_command = any(
        decorator.id == "apy_command"
        for decorator in ast_node.decorator_list
        if hasattr(decorator, "id")
    )

    metadata = {
        "type": "command" if is_command else "function",
        "visibility": visibility
    }
    if is_command:
        metadata["command"] = ast_node.name
    else:
        metadata["method"] = ast_node.name

    if len(function_source) > max_chunk_size:
        function_chunks = _split_into_chunks(function_source, max_chunk_size)
        for chunk in function_chunks:
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
    else:
        doc = Document(
            page_content=function_source,
            metadata=metadata
        )
        documents.append(doc)

    return documents

def _chunk_first_level_assign_node(ast_node, code_file_content, max_chunk_size):
    """
    Handles assignment statements at the first level of the AST.
    """
    documents = []
    assign_start_line = ast_node.lineno
    assign_end_line = ast_node.end_lineno
    assign_source = '\n'.join(code_file_content.splitlines()[assign_start_line-1:assign_end_line])

    metadata = {"type": "Assign"}

    if len(assign_source) > max_chunk_size:
        assign_chunks = _split_into_chunks(assign_source, max_chunk_size)
        for chunk in assign_chunks:
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
    else:
        doc = Document(
            page_content=assign_source,
            metadata=metadata
        )
        documents.append(doc)

    return documents

def _chunk_import_only_code_file_content(code_file_content, code_file_path, max_chunk_size):
    """
    Handles cases where the first-level nodes are only imports.
    """
    documents = []
    if code_file_path.endswith("__init__.py"):
        type = "__init__-file"
    else:
        type = "undefined"

    metadata = {"type": type}

    if len(code_file_content) > max_chunk_size:
        import_chunks = _split_into_chunks(code_file_content, max_chunk_size)
        for chunk in import_chunks:
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
    else:
        doc = Document(
            page_content=code_file_content,
            metadata=metadata
        )
        documents.append(doc)

    return documents

def _chunk_nodeless_code_file_content(code_file_content, code_file_path, max_chunk_size):
    """
    Handles cases where no top-level nodes are found in the AST.
    """
    documents = []
    if code_file_path.endswith("__init__.py"):
        type = "__init__-file"
    else:
        type = "undefined"

    metadata = {"type": type}

    if len(code_file_content) > max_chunk_size:
        nodeless_chunks = _split_into_chunks(code_file_content, max_chunk_size)
        for chunk in nodeless_chunks:
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
    else:
        doc = Document(
            page_content=code_file_content,
            metadata=metadata
        )
        documents.append(doc)

    return documents

def _handle_not_defined_case(ast_node, code_file_content, max_chunk_size):
    """
    Captures all lines corresponding to the given node and creates
    a Document with metadata for undefined type.
    """
    documents = []
    # Determine the start and end lines of the node
    start_line = ast_node.lineno
    end_line = ast_node.end_lineno

    # Extract the relevant lines from the code file content
    lines = code_file_content.splitlines()
    undefined_content = "\n".join(lines[start_line - 1:end_line])

    metadata = {"type": "undefined"}

    if len(undefined_content) > max_chunk_size:
        undefined_chunks = _split_into_chunks(undefined_content, max_chunk_size)
        for chunk in undefined_chunks:
            doc = Document(
                page_content=chunk,
                metadata=metadata
            )
            documents.append(doc)
    else:
        doc = Document(
            page_content=undefined_content,
            metadata=metadata
        )
        documents.append(doc)

    return documents


def _split_into_chunks(source, max_chunk_size):
    """Splits source content into smaller chunks of max_chunk_size."""
    lines = source.splitlines()
    chunks = []
    current_chunk = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1  # Add 1 for the newline character
        if current_size + line_size > max_chunk_size:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(line)
        current_size += line_size

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks
