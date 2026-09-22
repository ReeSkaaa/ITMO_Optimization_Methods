import numpy as np


def simplex_method(table, basis, free):
    n = table.shape[1] - 1  # количество переменных

    # выполняем действия пока не придем к конечному результату
    while True:
        # в последней строке первые числа - коэффициенты целевой функции
        # в последней строке последнее число справа = -Q
        # извлекаем последнюю строку без последнего элемента
        last_str = table[-1, :-1]
        # теперь необходимо найти самый минимальный элемент
        j = np.argmin(last_str)  # номер разрашаюшего столбца
        # мы не должны выбирать элемент если все элементы положительны
        if last_str[j] >= 0:
            print('Не можем выбрать столбец, все столбцы положительны')
            break
        # составим массив, где будет указан элемент из разрешающего столбца, отношение bi/aij, причем aij>0
        # если aij не будет удовлетворять условию, ставим бесконечность, она не будет выбрана
        # создаем изначально бесконечный массив
        arr = np.full(table.shape[0] - 1, np.inf)
        for r in range(table.shape[0] - 1):
            if table[r, j] > 0:
                arr[r] = table[r, -1] / table[r, j]
        if np.all(np.isinf(arr)):
            raise ValueError("Задача не ограничена")
        # выбираем разрешающую строку
        i = np.argmin(arr)
        # сохраним разреш элемент
        pivot = table[i, j]
        old_table = table.copy()

        # пересчет ведущей строки
        table[i, :] = old_table[i, :] / pivot
        # перерасчет разрешающего столбца
        table[:, j] = - old_table[:, j] / pivot

        #  установить значение самого pivot
        table[i, j] = 1 / pivot

        # пересчет оставшихся элементов
        for str_ in range(table.shape[0]):
            for col_ in range(table.shape[1]):
                # формула перессчета оставшихся элементов
                if str_ != i and col_ != j:
                    table[str_, col_] = old_table[str_, col_] - (old_table[str_, j] * old_table[i, col_] / pivot)

        # замена базиса i <-> j
        basis[i], free[j] = free[j], basis[i]

        print(table)
        print('------------------------\n')
    return table



def to_canonical(c, constraints, nonneg=None, sense='min'):
    # приведение к каноническому виду
    """

    :param c: коэффициенты целевой функции
    :param constraints: ограничения
    :param nonneg: какие переменные неотрицательны
    :param sense: цель минимизайия или максимизация
    :return:
    """
    n = len(c)
    if nonneg is None:
        nonneg = set(range(1, n + 1))

    c = [float(v) for v in c]

    # 1) max -> min
    if sense == 'max':
        c = [-v for v in c]
    # сколько нужно дополнительных переменных
    extra = sum(1 for (_, rel, _) in constraints if rel != '=')
    # итого переменных
    n_total = n + extra
    # необходима для разбития переменных на разности xk = (xk+)-(xk-)
    var_map = [('orig', k) for k in range(1, n + 1)]
    A, b = [], [] # матрица коэфф, матрица свободных членов
    n_extra = n # номер позиции дополнительной перемнной
    # перебираем ограничения
    for coeffs, rel, bi in constraints:
        row = [float(v) for v in coeffs] + [0.0] * (n_total - n)
        if rel == '<=':
            row[n_extra] = 1.0
            var_map.append(('slack',)) # новая переменная дополнительная
            n_extra += 1
        elif rel == '>=':
            row[n_extra] = -1.0
            var_map.append(('slack',))
            n_extra += 1
        A.append(row)
        b.append(float(bi))
    c = c + [0.0] * extra

    # 3) b_i < 0 -> строку на -1
    for i in range(len(b)):
        if b[i] < 0:
            A[i] = [-v for v in A[i]]
            b[i] = -b[i]

    # 4) свободные переменные -> x_k = x_k^+ - x_k^-
    free_vars = [k for k in range(1, n + 1) if k not in nonneg]
    for k in free_vars:
        j_k = next(j for j, vm in enumerate(var_map) if vm == ('orig', k))
        for i in range(len(A)):
            A[i].append(-A[i][j_k])
        c.append(-c[j_k])
        var_map[j_k] = ('plus', k)
        var_map.append(('minus', k))
    return np.array(A), np.array(b), np.array(c), var_map


def find_basis_columns(A, m, n):
    # пытаемся сразу найти единичный базис (столбец-e_i) для каждой строки
    basis_col = [None] * m
    used = set()
    for i in range(m):
        for j in range(n):
            if j in used:
                continue
            col = A[:, j]
            if col[i] == 1 and np.all(np.delete(col, i) == 0):
                basis_col[i] = j
                used.add(j)
                break
    return basis_col


def solve_general_lp(c, constraints, nonneg=None, sense='min'):
    A, b, c_full, var_map = to_canonical(c, constraints, nonneg, sense)
    m, n = A.shape

    # ищем итоговый единичный базис
    basis_col = find_basis_columns(A, m, n)
    basis = [None] * m
    used = set()
    for i in range(m):
        if basis_col[i] is not None:
            basis[i] = basis_col[i] + 1  # переменные нумеруем с 1
            used.add(basis_col[i] + 1)

    art_vars = []
    next_id = n + 1
    for i in range(m):
        if basis[i] is None:
            art_vars.append(next_id)
            basis[i] = next_id
            next_id += 1

    if art_vars:
        free = [k for k in range(1, next_id) if k not in basis]
        n_ext = next_id - 1
        A_ext = np.hstack([A, np.zeros((m, n_ext - n))])
        for i in range(m):
            if basis[i] in art_vars:
                A_ext[i, basis[i] - 1] = 1.0

        free_cols = [j - 1 for j in free]
        table = np.zeros((m + 1, len(free) + 1))
        table[:m, :-1] = A_ext[:, free_cols]
        table[:m, -1] = b

        c_basis = np.array([1.0 if v in art_vars else 0.0 for v in basis])
        c_free = np.array([1.0 if v in art_vars else 0.0 for v in free])
        table[-1, :-1] = c_free - c_basis @ table[:m, :-1]
        table[-1, -1] = -(c_basis @ b)

        basis_arr, free_arr = basis[:], free[:]
        table = simplex_method(table, basis_arr, free_arr)
        basis, free = basis_arr, free_arr

        W1 = -table[-1, -1]
        if abs(W1) > 1e-6:
            print(f"W'(x*) = {W1:.6f} > 0  =>  X = пусто, решений нет.")
            return None

        # если искусственная переменная осталась в базисе на уровне 0 -> выводим её
        for i in range(m):
            if basis[i] in art_vars:
                for j, v in enumerate(free):
                    if v not in art_vars and abs(table[i, j]) > 1e-9:
                        basis_arr, free_arr = basis, free
                        pivot = table[i, j]
                        old_table = table.copy()
                        table[i, :] = old_table[i, :] / pivot
                        table[:, j] = -old_table[:, j] / pivot
                        table[i, j] = 1 / pivot
                        for r in range(table.shape[0]):
                            for c_ in range(table.shape[1]):
                                if r != i and c_ != j:
                                    table[r, c_] = old_table[r, c_] - (old_table[r, j] * old_table[i, c_] / pivot)
                        basis[i], free[j] = free[j], basis[i]
                        break

        # убираем столбцы искусственных переменных из free
        keep = [j for j, v in enumerate(free) if v not in art_vars]
        table = np.hstack([table[:, keep], table[:, -1:]])
        free = [free[j] for j in keep]
    else:
        free = [k for k in range(1, n + 1) if k not in used]
        table = np.zeros((m + 1, len(free) + 1))
        table[:m, :-1] = A[:, [j - 1 for j in free]]
        table[:m, -1] = b

    c_basis = np.array([c_full[v - 1] for v in basis])
    c_free = np.array([c_full[v - 1] for v in free])
    table[-1, :-1] = c_free - c_basis @ table[:m, :-1]
    table[-1, -1] = -(c_basis @ table[:m, -1])

    table = simplex_method(table, basis, free)

    # решение
    x_full = np.zeros(n)
    for i in range(m):
        x_full[basis[i] - 1] = table[i, -1]

    n_orig = len(c)
    x = np.zeros(n_orig)
    for j, vm in enumerate(var_map):
        if vm[0] == 'orig':
            x[vm[1] - 1] = x_full[j]
        elif vm[0] == 'plus':
            x[vm[1] - 1] += x_full[j]
        elif vm[0] == 'minus':
            x[vm[1] - 1] -= x_full[j]

    W_value = float(np.dot(np.array(c, dtype=float), x))
    print('ИТОГ:')
    print('x* =', np.round(x, 6))
    print('W(x*) =', round(W_value, 6))
    return x, W_value


if __name__ == '__main__':
    # тут указываем что хотим сделать(min, max)
    target = 'min'
    # тут вводим все коэффициенты
    # целевая функция
    c = [3, 1, 4, 2]
    # ограничения
    constraints = [
        ([2, 1, 1, 0], '<=', 8),
        ([1, 0, 1, 1], '=', 5),
        ([0, 1, 0, 1], '>=', 4),
    ]
    solve_general_lp(c, constraints, nonneg={1, 2, 3, 4}, sense='min')
# если не указываете ккакие переменные неотрицательные то изначально верным считается x1....>=0