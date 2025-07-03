# define loss and get its derivative

import sympy
from sympy.functions.elementary.complexes import Abs

x, y = sympy.symbols("x y", real=True)

z = x + sympy.I * y
L_rel = (Abs(z**3 -1)**2).expand()
L_rel_x = sympy.diff(L_rel, x)
L_rel_y = sympy.diff(L_rel, y)

abs_z_squared = x**2 + y**2



L_unit = (x **2 + y ** 2 -1) ** 2
L_unit_x = sympy.diff(L_unit, x)
L_unit_y = sympy.diff(L_unit, y)
L_irr = ( 1/3 * (1 + abs_z_squared + abs_z_squared ** 2)  -1 ) ** 2
L_irr_x = sympy.diff(L_irr, x)
L_irr_y = sympy.diff(L_irr, y)


L = L_rel + L_unit + L_irr
L_x = sympy.diff(L, x)
L_y = sympy.diff(L, y)

loss = sympy.lambdify((x, y), L)
loss_x = sympy.lambdify((x, y), L_x)
loss_y = sympy.lambdify((x, y), L_y)

loss_rel = sympy.lambdify((x, y), L_rel)
loss_rel_x = sympy.lambdify((x, y), L_rel_x)
loss_rel_y = sympy.lambdify((x, y), L_rel_y)

loss_unit = sympy.lambdify((x, y), L_unit)
loss_unit_x = sympy.lambdify((x, y), L_unit_x)
loss_unit_y = sympy.lambdify((x, y), L_unit_y)  

loss_irr = sympy.lambdify((x, y), L_irr)
loss_irr_x = sympy.lambdify((x, y), L_irr_x)
loss_irr_y = sympy.lambdify((x, y), L_irr_y)
