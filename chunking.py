import ast
from langchain.schema import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
def chunk_pythoncode_and_add_metadata(code_files_content, code_files_path):
    chunks = [] 
    for code_file_content, code_file_path in zip(code_files_content, code_files_path):
        """
        Custom made python code splitter, algorithm iterates through child nodes of ast-tree(max child depth = 2)
        aims to have full body of methods along signature (+ can handle decorators) in a chunk and adds method specific metadata
        e.g visbility: public, _internal
            type: "class", "methods", "command"(CLI commands)
            source: 
        
        
        with the intend to use a filter when retrieving potentaion useful snippets. 
        
        
        """
        document_chunks = _generate_code_chunks_with_metadata(code_file_content, code_file_path)
        chunks.extend(document_chunks)   
    return chunks


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
                    "directory": "doc/"
                }
            ) 
            for chunk in text_splitter.split_text(text)
        ])
    return chunks


def _generate_code_chunks_with_metadata(code_file_content, code_file_path):
    """
    Custom Python Code Splitter
    chunks python file by length of func/method body
    aims to have one full method/function in a chunk and full body of a class, but cutting of when first method declaration is met
    able to handles decorators on methods
        
    Entry point method to process the Python file.
    It invokes the iterate_ast function.
    """
    documents = []
   #print(f"Processing file: {file_path}")
  
    _iterate_ast(code_file_content, documents, code_file_path)
    # Determine usage based on the file_path

    if code_file_path.startswith("kadi_apy"):
        directory = "kadi_apy/"
        if code_file_path.startswith("kadi_apy/lib/"):
            usage = "kadi_apy/lib/"
        elif code_file_path.startswith("kadi_apy/cli/"):
            usage = "kadi_apy/cli/"
        else:
            usage = "kadi_apy/top_level_file.py"
    else:
        directory = "undefined"
        usage = "undefined"
        
    # Add metadata-type "usage" to all documents
    for doc in documents:
        doc.metadata["source"] = code_file_path
        doc.metadata["directory"] = directory
        doc.metadata["usage"] = usage  # Add the determined usage metadata
        #print(doc)
    return documents

def _iterate_ast(code_file_content, documents, code_file_path):
    """
    Parses the AST of the given Python file and delegates
    handling to specific methods based on node types.
    """
    tree = ast.parse(code_file_content, filename=code_file_path)

    first_level_nodes = list(ast.iter_child_nodes(tree))

    # Check if there are no first-level nodes
    if not first_level_nodes:
        documents.extend(
            _chunk_nodeless_code_file_content(code_file_content, code_file_path))
        return

    all_imports = all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in first_level_nodes)

    if all_imports:
        # Handle the case where all first-level nodes are import statements
        documents.extend(
            _chunk_import_only_code_file_content(code_file_content, code_file_path)
        )
    else:
        # Handle the case where first-level nodes contain other types
        for first_level_node in ast.iter_child_nodes(tree):
            if isinstance(first_level_node, ast.ClassDef):
                documents.extend(
                    _handle_first_level_class(first_level_node, code_file_content)
                )
            elif isinstance(first_level_node, ast.FunctionDef):
                documents.extend(
                    _chunk_first_level_func_node(first_level_node, code_file_content)
                )
            elif isinstance(first_level_node, ast.Assign):
                documents.extend(
                    _chunk_first_level_assign_node(first_level_node, code_file_content)
                )
            else:
                if not isinstance(first_level_node, (ast.Import, ast.ImportFrom)):
                    documents.extend(
                        _handle_not_defined_case(first_level_node, code_file_content)
                    )

                


def _handle_first_level_class(ast_node , code_file_content):
    """
    Handles classes at the first level of the AST.
    """
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

    # Create and store Document for the class
    doc = Document(
        page_content=class_source,
        metadata=metadata
    )
    documents.append(doc)

    # Handle methods within the class
    for second_level_node in ast.iter_child_nodes(ast_node):
        if isinstance(second_level_node, ast.FunctionDef):
            method_start_line = (
                second_level_node.decorator_list[0].lineno
                if second_level_node.decorator_list else second_level_node.lineno
            )
            method_end_line = second_level_node.end_lineno
            method_source = '\n'.join(code_file_content.splitlines()[method_start_line-1:method_end_line])

            visibility = "internal" if second_level_node.name.startswith("_") else "public"

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


def _handle_not_defined_case(ast_node, code_file_content):
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

    # Metadata for undefined type
    metadata = {"type": "undefined"}

    # Create and return a Document
    doc = Document(
        page_content=undefined_content,
        metadata=metadata
    )
    documents.append(doc)
    return documents
    

def _chunk_first_level_func_node(ast_node, code_file_content):
    """
    Handles functions at the first level of the AST.
    """
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

    doc = Document(
        page_content=function_source,
        metadata=metadata
    )
    documents.append(doc)

    return documents



def _chunk_first_level_assign_node(ast_node, code_file_content):
    
    """
    Handles assignment statements at the first level of the AST.
    """
    documents = []
    assign_start_line = ast_node.lineno
    assign_end_line = ast_node.end_lineno
    assign_source = '\n'.join(code_file_content.splitlines()[assign_start_line-1:assign_end_line])

    # Create metadata without imports
    metadata = {"type": "Assign"}

    # Create and store Document for the assignment
    doc = Document(
        page_content=assign_source,
        metadata=metadata
    )
    documents.append(doc)

    return documents


    
def _chunk_import_only_code_file_content(code_file_content, code_file_path):
    """
    Handles cases where the first-level nodes are only imports.
    """
    documents = []
    if code_file_path.endswith("__init__.py"):
        type = "__init__-file"
    else:
        type = "undefined"

    # Create metadata without imports
    metadata = {"type": type}

    # Create and store a Document with the full source code
    doc = Document(
        page_content=code_file_content,
        metadata=metadata
    )
    documents.append(doc)
    return documents

def _chunk_nodeless_code_file_content(code_file_content, code_file_path):
    """
    Handles cases where no top-level nodes are found in the AST.
    """
    documents = []
    if code_file_path.endswith("__init__.py"):
        type = "__init__-file"
    else:
        type = "undefined"

    # Create metadata without imports
    metadata = {"type": type}

    # Create and store a Document with the full source code
    doc = Document(
        page_content=code_file_content,
        metadata=metadata
    )
    documents.append(doc)

    return documents




