import ast
import tokenize
import sys

def main():
    file = sys.argv[1]
    with open(file, "r") as source:
        t = ast.parse(source.read())
    
    with tokenize.open(file) as f:
        tokens = tokenize.generate_tokens(f.readline)
        print("Comments:")
        for token in tokens:
            if token.exact_type == 55: #COMMENT
                print("Line: ", token.start[0], ", Comment: ", token.string)

    impFinder = ImportFinder()
    impFinder.visit(t)
    impList = impFinder.getImports()
    print("Import (aliases): ", impList)

    funcFinder = FunctionFinder()
    funcFinder.visit(t)
    funcFinder.print()
    funclist = funcFinder.getFunctions()

    varFinder = VariableFinder()
    varFinder.setFunctions(funclist)
    varFinder.setImports(impList)
    varFinder.visit(t)
    varFinder.print()

    with open(file, "r") as f:
        bodydict = {}
        i = 1
        for line in f:
            for k, v in fdict.items:
                # compare line against function def
                if i == v[0]:

            i += 1


    # TODO: create dictionary of lists

class FunctionData:
    def __init__(self):
        self.args = []
        self.vars = []

class ImportFinder(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
    
    def visit_Import(self, node):
        for alias in node.names:
            if alias.asname is not None:
                self.imports.append(alias.asname)
            else:
                self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def getImports(self):
        return self.imports

class FunctionFinder(ast.NodeVisitor):
    def __init__(self):
        self.funcs = []
        self.fdict = {}

    # Top level function definitions
    def visit_FunctionDef(self, node):
        self.args = []
        self.funcs.append(node.name)
        for arg in node.args.args:
            if arg.arg is not "self":
                self.args.append(arg.arg)
        self.fdict[node.name] = (node.lineno, self.args)
        self.generic_visit(node)

    # In-function calls
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.funcs.append(node.func.id)
        if isinstance(node.func, ast.Attribute):
            self.funcs.append(node.func.value)
        self.generic_visit(node)


    def getFunctions(self):
        return self.funcs
    
    def print(self):
        print("Functions with arguments: ")
        for k, v in self.fdict.items():
            print(k, ":", v)


class VariableFinder(ast.NodeVisitor):
    def __init__(self):
        self.funcs = []
        self.vars = []
        self.imps = []
    
    def setFunctions(self, funcs):
        self.funcs = funcs
    
    def setImports(self, imps):
        self.imps = imps
    
    def visit_Name(self, node):
        # TODO: group variables by functions
        if node.id not in self.funcs and node.id is not "self" and node.id not in self.imps:
            self.vars.append((node.lineno, node.id))
        self.generic_visit(node)
    
    def print(self):
        # only print unique variables names
        print("Variables used in lines: \n", sorted(self.vars, key=lambda x: x[0]))
        print("Unique variables used: \n", set([y for (x,y) in self.vars]))

if __name__ == "__main__":
    main()
