import ast
import tokenize
import sys
import os
from io import StringIO

class ASTTraverser:

    def __init__(self):
        self.tdict = {}
        self.fdict = {}
        self.var_dict = {}

    def populate_dicts(self, file, fdict):
        self.fdict = fdict
        self.populate_tdict(file)
        self.populate_var_dict()

    def populate_tdict(self, file):
        with tokenize.open(file) as f:
            type_filter = [6, 55] #DEDENT, COMMENT
            fheadlines = []
            body = None
            for k, v in self.fdict.items():
                fheadlines.append((k, v[0]))
            i = 1
            in_body = False
            end_of_doc = False
            for line in f:
                for j in range(len(fheadlines)):
                    # compare line against function def
                    if fheadlines[j][1] <= i:
                        if fheadlines[j][1] == i:
                            in_body = True
                            break

                        try:
                            if i == fheadlines[j+1][1]:
                                in_body = False
                                self.tdict[fheadlines[j][0]] = body
                                body = None
                        except IndexError:
                            end_of_doc = True
                            continue
                if end_of_doc:
                    in_body = True
                if in_body:
                    if not body:
                        body = []
                    ts = tokenize.generate_tokens(StringIO(line).readline)
                    tokenized_line = []
                    try:
                        for token in ts:
                            if token.exact_type not in type_filter:
                                tokenized_line.append(token.string)
                        body.extend(tokenized_line)
                    except:
                        continue
                        #print("Error while tokenizing...")
                i += 1

    def populate_var_dict(self):
        """
        Creates and returns a dictionary with scheme "variable name: tokenized calling function body"
        \nParameters:
        fdict: func name, func parameters
        tdict: func name, tokenized func body
        """
        for fname, v in self.fdict.items():
            fparams = v[1:][0]
            #fname is the function name (key)
            #v[0] is the starting lineno
            #v[1:] are all func parameters
            #print(f"For {fname}, we have parameters {fparams}")
            for var in fparams:
                entry = []
                if var in self.var_dict:
                    entry = self.var_dict[var]
                if fname in self.tdict:
                    entry.append(self.tdict[fname])
                self.var_dict[var] = entry
    
    def get_var_dict(self):
        return self.var_dict


def main():
    file = sys.argv[1]
    # cwd = os.getcwd()
    with open(file, "r") as source:
        t = ast.parse(source.read())

    #impFinder = ImportFinder()
    #impFinder.visit(t)
    #impList = impFinder.getImports()
    #print("Import (aliases): ", impList)

    funcFinder = FunctionFinder()
    funcFinder.visit(t)
    #funcFinder.print()
    #funclist = funcFinder.getFunctions()
    fdict = funcFinder.getFdict()

    #varFinder = VariableFinder()
    #varFinder.setFunctions(funclist)
    #varFinder.setImports(impList)
    #varFinder.visit(t)
    #varFinder.print()

    trav = ASTTraverser()
    trav.populate_dicts(file, fdict)

    for k, v in trav.get_var_dict().items():
        print(f"For variable {k}, we have tokens of {len(v)} function(s).\n")

def populate_tdict(file, fdict):
    tdict = {}

    with tokenize.open(file) as f:
        type_filter = [5, 6, 55] #IDENT, DEDENT, COMMENT
        fheadlines = []
        body = None
        for k, v in fdict.items():
            fheadlines.append((k, v[0]))
        i = 1
        in_body = False
        end_of_doc = False
        for line in f:
            for j in range(len(fheadlines)):
                # compare line against function def
                if fheadlines[j][1] <= i:
                    if fheadlines[j][1] == i:
                        in_body = True
                        break

                    try:
                        if i == fheadlines[j+1][1]:
                            in_body = False
                            tdict[fheadlines[j][0]] = body
                            body = None
                    except IndexError:
                        end_of_doc = True
                        continue
            if end_of_doc:
                in_body = True
            if in_body:
                if not body:
                    body = []
                ts = tokenize.generate_tokens(StringIO(line).readline)
                tokenized_line = []
                try:
                    for token in ts:
                        if token.exact_type not in type_filter:
                            tokenized_line.append(token.string)
                except:
                    print("Error tokenizing")
                    pass
                body.extend(tokenized_line)
            i += 1
    return tdict

def populate_var_dict(fdict, tdict):
    """
    Creates and returns a dictionary with scheme "variable name: tokenized calling function body"
    \nParameters:
        fdict: func name, func parameters
        tdict: func name, tokenized func body
    """
    var_dict = {}
    for fname, v in fdict.items():
        fparams = v[1:][0]
        #fname is the function name (key)
        #v[0] is the starting lineno
        #v[1:] are all func parameters
        print(f"For {fname}, we have parameters {fparams}")
        for var in fparams:
            entry = []
            if var in var_dict:
                entry = var_dict[var]
            if fname in tdict:
                entry.append(tdict[fname])
            var_dict[var] = entry
    return var_dict

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
    
    def getFdict(self):
        return self.fdict
    
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
    
    def getUniqueVars(self):
        return set([y for (x,y) in self.vars])
    
    def print(self):
        # only print unique variables names
        print("Variables used in lines: \n", sorted(self.vars, key=lambda x: x[0]))
        print("Unique variables used: \n", self.getUniqueVars())

if __name__ == "__main__":
    main()
