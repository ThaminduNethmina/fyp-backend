import ast
import re
import javalang

# Java Cleaner
def clean_java_code(code):
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    code = re.sub(r'^\s*import\s+.*;', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*package\s+.*;', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()

# Python Cleaner
def clean_python_code(code):
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    code = re.sub(r'^\s*import\s+.*', '', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*from\s+.*import.*', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()


def clean_code(code, lang):
    if lang == 'python':
        return clean_python_code(code)
    else:
        return clean_java_code(code)
    

def get_python_features(code):
    try:
        tree = ast.parse(code)
    except:
        return [0, 0, 0, 0, 0]

    max_depth = 0
    branch_count = 0
    has_recursion = 0
    has_log_math = 0
    has_sort = 0

    current_functions = []

    class DepthVisitor(ast.NodeVisitor):
        def __init__(self):
            self.max_depth = 0
            self.current_depth = 0

        def visit_For(self, node):
            self.current_depth += 1
            self.max_depth = max(self.max_depth, self.current_depth)
            self.generic_visit(node)
            self.current_depth -= 1

        def visit_While(self, node):
            self.current_depth += 1
            self.max_depth = max(self.max_depth, self.current_depth)
            self.generic_visit(node)
            self.current_depth -= 1

        def visit_ListComp(self, node):
            self.current_depth += len(node.generators)
            self.max_depth = max(self.max_depth, self.current_depth)
            self.generic_visit(node)
            self.current_depth -= len(node.generators)

    depth_visitor = DepthVisitor()
    depth_visitor.visit(tree)
    max_depth = depth_visitor.max_depth

    for node in ast.walk(tree):
        # Branch Counting
        if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ListComp)):
            branch_count += 1

        # Recursion & Sort Detection
        if isinstance(node, ast.FunctionDef):
            current_functions.append(node.name)

        if isinstance(node, ast.Call):
            # Recursion
            if isinstance(node.func, ast.Name) and node.func.id in current_functions:
                has_recursion = 1
            # Sort Detection: sorted(arr)
            if isinstance(node.func, ast.Name) and node.func.id == 'sorted':
                has_sort = 1
            # Sort Detection: arr.sort()
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'sort':
                has_sort = 1

        # Logarithmic Math Detection
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.RShift, ast.Mult, ast.LShift)):
                has_log_math = 1
        if isinstance(node, ast.AugAssign):
             if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.RShift, ast.Mult, ast.LShift)):
                has_log_math = 1

    # Return 5 features
    return [max_depth, branch_count, has_recursion, has_log_math, has_sort]

def get_java_features(code):
    try:
        if "class " not in code:
             tokens = javalang.tokenizer.tokenize("class Dummy { " + code + " }")
        else:
             tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
    except:
        return [0, 0, 0, 0, 0]

    real_max_depth = 0
    branch_count = 0
    has_recursion = 0
    has_log_math = 0
    has_sort = 0

    # Max Depth
    for path, node in tree.filter(javalang.tree.ForStatement):
        current = sum(1 for p in path if isinstance(p, (javalang.tree.ForStatement, javalang.tree.WhileStatement, javalang.tree.DoStatement)))
        real_max_depth = max(real_max_depth, current + 1)

    for path, node in tree.filter(javalang.tree.WhileStatement):
        current = sum(1 for p in path if isinstance(p, (javalang.tree.ForStatement, javalang.tree.WhileStatement, javalang.tree.DoStatement)))
        real_max_depth = max(real_max_depth, current + 1)

    # Branch Count
    for path, node in tree.filter(javalang.tree.IfStatement):
        branch_count += 1

    # Recursion & Sorting
    methods = [node.name for path, node in tree.filter(javalang.tree.MethodDeclaration)]
    for path, node in tree.filter(javalang.tree.MethodInvocation):
        if node.member in methods:
            has_recursion = 1
        if node.member == 'sort':
            has_sort = 1

    # AST-Based Log Math
    for path, node in tree.filter(javalang.tree.BinaryOperation):
        if node.operator in ['/', '*', '>>', '<<', '>>>']:
            has_log_math = 1

    for path, node in tree.filter(javalang.tree.Assignment):
        if node.type in ['/=', '*=', '>>=', '<<=', '>>>=']:
            has_log_math = 1

    return [real_max_depth, branch_count, has_recursion, has_log_math, has_sort]