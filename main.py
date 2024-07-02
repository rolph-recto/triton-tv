from typing import Optional, Union
import z3

# TYPES

sort_map = {}
def get_sort(s: str):
    global sort_map
    if s not in sort_map:
        sort_map[s] = z3.DeclareSort(s)

    return sort_map[s]

class Type:
    def as_z3(self) -> z3.Sort:
        assert False, "unimplemented"

class BaseType(Type):
    pass

class IntType(BaseType):
    def __init__(self):
        pass

    def as_z3(self):
        return z3.IntSort()

    def __str__(self):
        return "int"

    def __eq__(self, other):
        return isinstance(other, IntType)

class FloatType(BaseType):
    def __init__(self):
        pass

    def as_z3(self):
        return z3.RealSort()

    def __str__(self):
        return "float"

    def __eq__(self, other):
        return isinstance(other, FloatType)

class PointerType(BaseType):
    def __init__(self, base: BaseType):
        self.base = base

    # treat pointers as raw addresses (ints)
    def as_z3(self):
        return z3.IntSort()

    def __str__(self):
        return f"{self.base}*"

    def __eq__(self, other):
        return isinstance(other, PointerType) and self.base == other.base 

class TensorType(Type):
    def __init__(self, base: BaseType, shape: list[int]):
        self.shape = shape
        self.base = base

    def as_z3(self):
        return get_sort(str(self))

    def lift_to_z3(self) -> list[z3.Sort]:
        return ([z3.IntSort() for _ in range(len(self.shape))] + [self.base.as_z3()])

    def __str__(self):
        return f"{self.base}{self.shape}"

    def __eq__(self, other):
        return isinstance(other, TensorType) and self.base == other.base  and self.shape == other.shape

# AST

class AST:
    pass

class Expr(AST):
    pass

class Stmt(AST):
    pass

class Op(AST):
    pass

class OpStmt(Stmt):
    def __init__(self, id: Union[str,tuple[str,str]] , op: Op):
        self.id = id
        self.op = op

class Pointer(Expr):
    def __init__(self):
        pass

class Splat(Op):
    def __init__(self, ptr: str, shape: list[int]):
        self.ptr = ptr
        self.shape = shape

class MakeRange(Op):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

class AddPtr(Op):
    def __init__(self, ptr: str, offset: str):
        self.ptr = ptr
        self.offset = offset

class Load(Op):
    def __init__(self, ptr: str):
        self.ptr = ptr

class Store(Op):
    def __init__(self, ptr: str, val: str):
        self.ptr = ptr
        self.val = val

class Broadcast(Op):
    def __init__(self, val: str, shape: list[int]):
        self.val = val
        self.shape = shape

class ExpandDims(Op):
    def __init__(self, val: str, dim: int):
        self.val = val
        self.dim = dim

class Transpose(Op):
    def __init__(self, val: str, order: list[int]):
        self.val = val
        self.order = order

class Dot(Op):
    def __init__(self, a: str, b: str, c: Optional[str]=None):
        self.a = a
        self.b = b
        self.c = c

class Split(Op):
    def __init__(self, val: str):
        self.val = val

# INTERPRETER

class Value:
    def __init__(self, z3val, type: Type):
        self.z3val = z3val
        self.type = type

class Z3InterpreterContext:
    def __init__(self, solver: z3.Solver, store={}):
        self.solver = solver
        self.solver.set(":unsat_core", True)
        self.assertions = {}

        # map from variables to values
        self.store = store

        # global memory
        self.mem_version = 0
        self.mem = z3.Function(f"__mem{self.mem_version}__", z3.IntSort(), z3.IntSort())

        # assume that we won't have more ten 10-dimensional tensors
        self.var = [z3.Int(f"$i{i}") for i in range(10)]

        self.set_math()

    def range_constraints(self, shape: list[int]):
        range_constrs = []
        for i in range(len(shape)):
            range_constrs.append(self.var[i] >= 0)
            range_constrs.append(self.var[i] < shape[i])

        return z3.And(*range_constrs)

    def tensor_type(self, domain: list[int], codomain: z3.Sort) -> list[z3.Sort]:
        return [z3.IntSort() for _ in range(len(domain))] + [codomain]

    def set_math(self):
        self.math = {}
        exp = z3.Function("exp", z3.IntSort(), z3.IntSort())
        self.solver.add(
            z3.ForAll(self.var[:2],
                exp(self.var[0] + self.var[1]) == exp(self.var[0]) * exp(self.var[1])))

        self.solver.add(
            z3.ForAll(self.var[:2],
                exp(self.var[0] - self.var[1]) == exp(self.var[0]) / exp(self.var[1])))

        self.solver.add(exp(0) == 1)

        self.math["exp"] = exp

    def add_assert(self, prop, id_str: str):
        id = z3.Bool(f"$PROP_{id_str}")
        self.solver.assert_and_track(prop, id)
        self.assertions[id] = prop

    def interpret(self, stmts: list[OpStmt]):
        for stmt in stmts:
            if isinstance(stmt.op, MakeRange):
                assert stmt.op.start < stmt.op.end

                n = stmt.op.end - stmt.op.start
                f = z3.Function(stmt.id, z3.IntSort(), z3.IntSort())

                # forall 0 <= v < n. f(v) = v + start
                fdef = \
                    z3.ForAll([self.var[0]],
                        z3.Implies(
                            z3.And(self.var[0] >= 0, self.var[0] < n),
                            f(self.var[0]) == self.var[0] + stmt.op.start))

                self.add_assert(fdef, stmt.id)
                self.store[stmt.id] = Value(f, TensorType(IntType(), [n]))

            elif isinstance(stmt.op, Splat):
                ptr = self.store[stmt.op.ptr]

                assert isinstance(ptr.type, PointerType)

                n = len(stmt.op.shape)
                f = z3.Function(stmt.id, *([z3.IntSort() for _ in range(n)] + [ptr.type.as_z3()]))

                # forall 0 <= v1 < s1, ..., 0 <= vn < sn. f(v1, ..., vn) == ptr
                fdef = z3.ForAll(self.var[:n],
                    z3.Implies(
                        self.range_constraints(stmt.op.shape),
                        f(*[self.var[:n]]) == ptr.z3val))

                self.add_assert(fdef, stmt.id)
                self.store[stmt.id] = Value(f, TensorType(ptr.type, stmt.op.shape))

            elif isinstance(stmt.op, AddPtr):
                ptr = self.store[stmt.op.ptr]
                offset = self.store[stmt.op.offset]

                assert isinstance(ptr.type, TensorType)
                assert isinstance(ptr.type.base, PointerType)
                assert isinstance(offset.type, TensorType)
                assert isinstance(offset.type.base, IntType)
                assert ptr.type.shape == offset.type.shape

                n = len(ptr.type.shape)
                f = z3.Function(stmt.id, *ptr.type.lift_to_z3())

                # forall 0 <= v1 < s1, ..., 0 <= vn < sn. f(v1, ..., vn) == ptr(v1, ..., vn) + offset(v1, ..., vn)
                fdef = z3.ForAll(self.var[:n],
                    z3.Implies(
                        self.range_constraints(ptr.type.shape),
                        f(*[self.var[:n]]) == ptr.z3val(*self.var[:n]) + offset.z3val(*self.var[:n])))

                self.add_assert(fdef, stmt.id)
                self.store[stmt.id] = Value(f, ptr.type)

            elif isinstance(stmt.op, Load):
                ptr = self.store[stmt.op.ptr]

                assert isinstance(ptr.type, TensorType)
                assert isinstance(ptr.type.base, PointerType)

                n = len(ptr.type.shape)
                f = z3.Function(stmt.id, *self.tensor_type(ptr.type.shape, ptr.type.base.base.as_z3()))

                # forall 0 <= v1 < s1, ..., 0 <= vn < sn. f(v1, ..., vn) == mem(ptr(v1, ..., vn))
                fdef = z3.ForAll(self.var[:n],
                    z3.Implies(
                        self.range_constraints(ptr.type.shape),
                        f(*[self.var[:n]]) == self.mem(ptr.z3val(*self.var[:n]))))

                self.add_assert(fdef, stmt.id)
                self.store[stmt.id] = Value(f, TensorType(ptr.type.base.base, ptr.type.shape))

            elif isinstance(stmt.op, Store):
                ptr = self.store[stmt.op.ptr]
                val = self.store[stmt.op.val]

                assert isinstance(ptr.type, TensorType)
                assert isinstance(ptr.type.base, PointerType)
                assert isinstance(val.type, TensorType)
                assert ptr.type.base.base == val.type.base
                assert ptr.type.shape == val.type.shape

                n = len(ptr.type.shape)
                self.mem_version += 1
                new_mem = z3.Function(f"__mem{self.mem_version}__", z3.IntSort(), z3.IntSort())

                # forall x, 0 <= v1 < s1, ..., 0 <= vn < sn.
                #   new_mem(x) == ite(x == ptr(v1, ..., vn), val(v1, ..., vn), mem(x))
                memdef = z3.ForAll(self.var[:n+1],
                    z3.Implies(
                        self.range_constraints(ptr.type.shape),
                        new_mem(self.var[n]) ==
                            z3.If(self.var[n] == ptr.z3val(*self.var[:n]),
                                val.z3val(*self.var[:n]),
                                self.mem(self.var[n]))))

                self.add_assert(memdef, stmt.id)
                self.mem = new_mem

            elif isinstance(stmt.op, Broadcast):
                val = self.store[stmt.op.val]

                assert isinstance(val.type, TensorType)
                assert len(val.type.shape) == len(stmt.op.shape)

                n = len(val.type.shape)
                broadcast_dim = None
                for i, d in enumerate(stmt.op.shape):
                    if val.type.shape[i] != d:
                        if broadcast_dim is None:
                            broadcast_dim = i

                        else:
                            assert f"multiple broadcast dims: {broadcast_dim} and {i}"

                # output shape is the same as input shape
                if broadcast_dim is None:
                    self.store[stmt.id] = val

                else:
                    indices = self.var[:n][:]
                    indices[broadcast_dim] = 0

                    f = z3.Function(stmt.id, *self.tensor_type(stmt.op.shape, val.type.base.as_z3()))

                    # let vi be the broadcast dimension; then
                    # forall v1, ..., vn. f(v1, ..., vn) == val(v1, ..., 0, ..., vn)
                    fdef = z3.ForAll(self.var[:n],
                        z3.Implies(
                            self.range_constraints(stmt.op.shape),
                            f(*self.var[:n]) == val.z3val(*indices)))

                    self.add_assert(fdef, stmt.id)
                    self.store[stmt.id] = Value(f, TensorType(val.type.base, stmt.op.shape))

            elif isinstance(stmt.op, Transpose):
                val = self.store[stmt.op.val]

                assert isinstance(val.type, TensorType)
                assert len(val.type.shape) == len(stmt.op.order)

                n = len(val.type.shape)
                indices: list[z3.ArithRef] = []
                for i in range(n):
                    assert 0 <= stmt.op.order[i] and stmt.op.order[i] < n
                    indices.append(self.var[stmt.op.order[i]])

                output_shape = [val.type.shape[i] for i in stmt.op.order]
                f = z3.Function(stmt.id, self.tensor_type(output_shape, val.type.base.as_z3()))

                # let pi be the permutation induced by order; then
                # forall v1, ..., vn. f(v1, ..., vn) == val(v_pi(1), ..., v_pi(n))
                fdef = z3.ForAll(self.var[:n],
                    z3.Implies(
                        self.range_constraints(output_shape),
                        f(*self.var[:n]) == val.z3val(*indices)))

                self.add_assert(fdef, stmt.id)
                self.store[stmt.id] = Value(f, TensorType(val.type.base, output_shape))

            elif isinstance(stmt.op, ExpandDims):
                val = self.store[stmt.op.val]

                assert isinstance(val.type, TensorType)

                n = len(val.type.shape)
                nout = n + 1
                assert 0 <= stmt.op.dim and stmt.op.dim <= n

                output_shape = val.type.shape[:]
                output_shape.insert(stmt.op.dim, 1)

                indices = self.var[:stmt.op.dim] + self.var[stmt.op.dim+1:nout]

                f = z3.Function(stmt.id, self.tensor_type(output_shape, val.type.base.as_z3()))

                # let i be the newly added dim; then
                # forall v1, ..., vn. f(v1, ..., vn) == val(v1, ..., v{i-1}, v{i+1}, ..., vn)
                fdef = z3.ForAll(self.var[:nout],
                    z3.Implies(
                        self.range_constraints(output_shape),
                        f(*self.var[:nout]) == val.z3val(*indices)))
                
                self.add_assert(fdef, stmt.id)
                self.store[stmt.id] = Value(f, TensorType(val.type.base, output_shape))

            elif isinstance(stmt.op, Dot):
                a = self.store[stmt.op.a]
                b = self.store[stmt.op.b]
                c = self.store[stmt.op.c] if stmt.op.c is not None else None

                assert isinstance(a.type, TensorType)
                assert isinstance(a.type.base, BaseType)
                assert isinstance(b.type, TensorType)
                assert isinstance(b.type.base, BaseType)
                assert a.type.base == b.type.base
                assert len(a.type.shape) == 2 and len(b.type.shape) == 2
                assert a.type.shape[1] == b.type.shape[0]

                output_shape = [a.type.shape[0], b.type.shape[1]]

                fname = stmt.id if c is None else f"{stmt.id}_2"
                f = z3.Function(fname, z3.IntSort(), z3.IntSort(), a.type.base.as_z3())

                # forall x, y. f(x, y) == a(x,0)*b(0,y) + ... + a(x,k)*b(k,y)
                sum_def = [a.z3val(self.var[0], i)*b.z3val(i, self.var[1]) for i in range(a.type.shape[1])]
                fdef = z3.ForAll(self.var[:2],
                    z3.Implies(
                        self.range_constraints(output_shape),
                        f(*self.var[:2]) == z3.Sum(*sum_def)))

                self.add_assert(fdef, fname)

                if c is None:
                    self.store[stmt.id] = f

                else:
                    assert isinstance(c.type, TensorType)
                    assert a.type.base == c.type.base
                    assert c.type.shape == output_shape

                    g = z3.Function(stmt.id, z3.IntSort(), z3.IntSort(), c.type.base.as_z3())
                    # forall x, y. g(x,y) == f(x,y) + c(x,y)
                    gdef = z3.ForAll(self.var[:2],
                        z3.Implies(
                            self.range_constraints(output_shape),
                            g(*self.var[:2]) == f(*self.var[:2]) + c.z3var(*self.var[:2])))

                    self.add_assert(gdef, stmt.id)
                    self.store[stmt.id] = Value(g, TensorType(a.type.base, output_shape))

            elif isinstance(stmt.op, Split):
                val = self.store[stmt.op.val]

                assert isinstance(val.type, TensorType)
                assert val.type.shape[-1] == 2
                assert isinstance(stmt.id, tuple)
                assert len(stmt.id) == 2

                n = len(val.type.shape)
                f1 = z3.Function(stmt.id[0], *self.tensor_type(val.type.shape[:-1], val.type.base.as_z3()))
                f1def = z3.ForAll(self.var[:n-1],
                    z3.Implies(
                        self.range_constraints(val.type.shape[:-1]),
                        f1(*self.var[:n-1]) == val.z3val(*(self.var[:n-1] + [0]))))

                self.add_assert(f1def, stmt.id[0])
                self.store[stmt.id[0]] = Value(f1, TensorType(val.type.base, val.type.shape[:-1]))

                f2 = z3.Function(stmt.id[1], *self.tensor_type(val.type.shape[:-1], val.type.base.as_z3()))
                f2def = z3.ForAll(self.var[:n-1],
                    z3.Implies(
                        self.range_constraints(val.type.shape[:-1]),
                        f2(*self.var[:n-1]) == val.z3val(*(self.var[:n-1] + [1]))))

                self.add_assert(f2def, stmt.id[1])
                self.store[stmt.id[1]] = Value(f2, TensorType(val.type.base, val.type.shape[:-1]))

            else:
                assert False, f"Unknown statement type {stmt}"

def main():
    store = {
        "a": Value(z3.Int("a"), PointerType(IntType()))
    }

    ctx = Z3InterpreterContext(z3.Solver(), store)

    ctx.interpret([
        OpStmt("i1", MakeRange(0, 10)),
        OpStmt("i2", Splat("a", [10])),
        OpStmt("i3", AddPtr("i2", "i1")),
        OpStmt("i4", Store("i3", "i1")),
        OpStmt("i5", Load("i3")),
        OpStmt("i6", ExpandDims("i1", 1)),
        OpStmt("i7", Broadcast("i6", [10, 2])),
        OpStmt("i8", Transpose("i7", [1, 0])),
        OpStmt(("i9_1", "i9_2"), Split("i7")),
    ])

    i5n = len(ctx.store["i5"].type.shape)
    i8 = ctx.store["i8"].z3val

    # check invariants
    ctx.solver.add(
        z3.Not(z3.And(
            # i1 == i5
            z3.ForAll(ctx.var[:i5n],
                z3.Implies(
                    ctx.range_constraints(ctx.store["i5"].type.shape),
                    ctx.store["i5"].z3val(*ctx.var[:i5n]) == ctx.store["i1"].z3val(*ctx.var[:i5n]))),

            # i1 == i9_1
            z3.ForAll(ctx.var[:i5n],
                z3.Implies(
                    ctx.range_constraints(ctx.store["i9_1"].type.shape),
                    ctx.store["i9_1"].z3val(*ctx.var[:i5n]) == ctx.store["i1"].z3val(*ctx.var[:i5n]))),

            # i9_1 == i9_2
            z3.ForAll(ctx.var[:i5n],
                z3.Implies(
                    ctx.range_constraints(ctx.store["i9_1"].type.shape),
                    ctx.store["i9_1"].z3val(*ctx.var[:i5n]) == ctx.store["i9_2"].z3val(*ctx.var[:i5n]))),

            # forall y. i8[0,y] == i8[1,y]
            z3.ForAll([ctx.var[0], ctx.var[1]],
                z3.Implies(
                    ctx.range_constraints(ctx.store["i8"].type.shape),
                    i8(0, ctx.var[1]) == i8(1, ctx.var[1]))))))

    print(ctx.solver.sexpr())
    print("solving...")
    if ctx.solver.check() == z3.sat:
        print("Satisfiable")
        print(ctx.solver.model())

    else:
        print("Unsatisfiable")
        print("Unsat core:")
        for prop in ctx.solver.unsat_core():
            print(prop, ":", ctx.assertions[prop])

if __name__ == "__main__":
    main()
