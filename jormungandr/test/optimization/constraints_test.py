from jormungandr.autodiff import Variable, VariableMatrix


def test_equality_constraint_boolean_comparisons():
    args = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]

    # double-Variable
    for lhs, rhs in args:
        assert bool(float(lhs) == Variable(rhs)) == (lhs == rhs)

    # double-VariableMatrix
    for lhs, rhs in args:
        assert bool(float(lhs) == VariableMatrix([[rhs]])) == (lhs == rhs)

    # Variable-double
    for lhs, rhs in args:
        assert bool(Variable(lhs) == float(rhs)) == (lhs == rhs)

    # Variable-Variable
    for lhs, rhs in args:
        assert bool(Variable(lhs) == Variable(rhs)) == (lhs == rhs)

    # Variable-VariableMatrix
    for lhs, rhs in args:
        assert bool(Variable(lhs) == VariableMatrix([[rhs]])) == (lhs == rhs)

    # VariableMatrix-double
    for lhs, rhs in args:
        assert bool(VariableMatrix([[lhs]]) == float(rhs)) == (lhs == rhs)

    # VariableMatrix-Variable
    for lhs, rhs in args:
        assert bool(VariableMatrix([[lhs]]) == Variable(rhs)) == (lhs == rhs)

    # VariableMatrix-VariableMatrix
    for lhs, rhs in args:
        assert bool(VariableMatrix([[lhs]]) == VariableMatrix([[rhs]])) == (lhs == rhs)


# For the purposes of optimization, a < constraint is treated the same as a <=
# constraint
def test_inequality_constraint_boolean_comparisons():
    args = [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0)]

    # double-Variable
    for lhs, rhs in args:
        assert bool(float(lhs) < Variable(rhs)) == (lhs <= rhs)
        assert bool(float(lhs) <= Variable(rhs)) == (lhs <= rhs)
        assert bool(float(lhs) > Variable(rhs)) == (lhs >= rhs)
        assert bool(float(lhs) >= Variable(rhs)) == (lhs >= rhs)

    # double-VariableMatrix
    for lhs, rhs in args:
        assert bool(float(lhs) < VariableMatrix([[rhs]])) == (lhs <= rhs)
        assert bool(float(lhs) <= VariableMatrix([[rhs]])) == (lhs <= rhs)
        assert bool(float(lhs) > VariableMatrix([[rhs]])) == (lhs >= rhs)
        assert bool(float(lhs) >= VariableMatrix([[rhs]])) == (lhs >= rhs)

    # Variable-Variable
    for lhs, rhs in args:
        assert bool(Variable(lhs) < Variable(rhs)) == (lhs <= rhs)
        assert bool(Variable(lhs) <= Variable(rhs)) == (lhs <= rhs)
        assert bool(Variable(lhs) > Variable(rhs)) == (lhs >= rhs)
        assert bool(Variable(lhs) >= Variable(rhs)) == (lhs >= rhs)

    # Variable-VariableMatrix
    for lhs, rhs in args:
        assert bool(Variable(lhs) < VariableMatrix([[rhs]])) == (lhs <= rhs)
        assert bool(Variable(lhs) <= VariableMatrix([[rhs]])) == (lhs <= rhs)
        assert bool(Variable(lhs) > VariableMatrix([[rhs]])) == (lhs >= rhs)
        assert bool(Variable(lhs) >= VariableMatrix([[rhs]])) == (lhs >= rhs)

    # VariableMatrix-double
    for lhs, rhs in args:
        assert bool(VariableMatrix([[lhs]]) < float(rhs)) == (lhs <= rhs)
        assert bool(VariableMatrix([[lhs]]) <= float(rhs)) == (lhs <= rhs)
        assert bool(VariableMatrix([[lhs]]) > float(rhs)) == (lhs >= rhs)
        assert bool(VariableMatrix([[lhs]]) >= float(rhs)) == (lhs >= rhs)

    # VariableMatrix-Variable
    for lhs, rhs in args:
        assert bool(VariableMatrix([[lhs]]) < Variable(rhs)) == (lhs <= rhs)
        assert bool(VariableMatrix([[lhs]]) <= Variable(rhs)) == (lhs <= rhs)
        assert bool(VariableMatrix([[lhs]]) > Variable(rhs)) == (lhs >= rhs)
        assert bool(VariableMatrix([[lhs]]) >= Variable(rhs)) == (lhs >= rhs)

    # VariableMatrix-VariableMatrix
    for lhs, rhs in args:
        assert bool(VariableMatrix([[lhs]]) < VariableMatrix([[rhs]])) == (lhs <= rhs)
        assert bool(VariableMatrix([[lhs]]) <= VariableMatrix([[rhs]])) == (lhs <= rhs)
        assert bool(VariableMatrix([[lhs]]) > VariableMatrix([[rhs]])) == (lhs >= rhs)
        assert bool(VariableMatrix([[lhs]]) >= VariableMatrix([[rhs]])) == (lhs >= rhs)
