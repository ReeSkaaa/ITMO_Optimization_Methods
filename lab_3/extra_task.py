import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 6.5))

# --- График 1: 3D-визуализация сферы и значений целевой функции ---
ax1 = fig.add_subplot(121, projection='3d')

u = np.linspace(0, 2 * np.pi, 80)
v = np.linspace(0, np.pi, 40)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones_like(u), np.cos(v))

# Значения целевой функции f(x, y, z) = x^2 + 2y^2 + 4z^2 на сфере
F_val = X**2 + 2 * Y**2 + 4 * Z**2

norm = plt.Normalize(F_val.min(), F_val.max())
colors = plt.cm.coolwarm(norm(F_val))

surf = ax1.plot_surface(
    X, Y, Z, facecolors=colors, alpha=0.75,
    linewidth=0, antialiased=True, shade=False
)

# Цветовая шкала
mappable = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
mappable.set_array(F_val)
fig.colorbar(mappable, ax=ax1, shrink=0.65, pad=0.1, label='Значение f(x, y, z)')

# Отмечаем 6 точек-кандидатов
ax1.scatter([1, -1], [0, 0], [0, 0], color='blue', s=120, edgecolors='black', zorder=10, label='Мин: (±1, 0, 0), f=1')
ax1.scatter([0, 0], [1, -1], [0, 0], color='gold', s=120, edgecolors='black', zorder=10, label='Седло: (0, ±1, 0), f=2')
ax1.scatter([0, 0], [0, 0], [1, -1], color='red', s=120, edgecolors='black', zorder=10, label='Макс: (0, 0, ±1), f=4')

ax1.set_title('1. Целевая функция на сфере $x^2 + y^2 + z^2 = 1$')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_xlim([-1.3, 1.3])
ax1.set_ylim([-1.3, 1.3])
ax1.set_zlim([-1.3, 1.3])
ax1.legend(loc='upper left', fontsize=8.5)

# --- График 2: 2D-сечение (y = 0) — касание линий уровня и коллинеарность градиентов ---
ax2 = fig.add_subplot(122)

grid = np.linspace(-2.2, 2.2, 400)
Xg, Zg = np.meshgrid(grid, grid)
Fg = Xg**2 + 4 * Zg**2

# Линии уровня f(x, 0, z) = c
levels = [0.5, 1.0, 2.0, 4.0, 6.0]
cs = ax2.contour(Xg, Zg, Fg, levels=levels, colors=['#4a90e2', 'blue', 'gray', 'red', '#d9534f'], linestyles='--')
ax2.clabel(cs, inline=True, fontsize=9, fmt='f = %.1f')

# Ограничение: сечение сферы x^2 + z^2 = 1
theta = np.linspace(0, 2 * np.pi, 300)
ax2.plot(np.cos(theta), np.sin(theta), 'k-', lw=2.5, label='Ограничение: $x^2 + z^2 = 1$')

# Точки касания (минимумы и максимумы)
ax2.scatter([1, -1], [0, 0], color='blue', s=100, zorder=5, label='Касание при f = 1 (Мин)')
ax2.scatter([0, 0], [1, -1], color='red', s=100, zorder=5, label='Касание при f = 4 (Макс)')

# Векторы градиентов в точке максимума (0, 1): grad(f) = (0, 8), grad(h) = (0, 2)
ax2.quiver(0, 1, 0, 0.85, angles='xy', scale_units='xy', scale=1, color='red', width=0.008, label=r'$\nabla f$ и $\nabla h$ (коллинеарны)')
ax2.quiver(0, 1, 0, 0.45, angles='xy', scale_units='xy', scale=1, color='black', width=0.006)

# Векторы градиентов в точке минимума (1, 0): grad(f) = (2, 0), grad(h) = (2, 0)
ax2.quiver(1, 0, 0.65, 0, angles='xy', scale_units='xy', scale=1, color='blue', width=0.008)

ax2.set_title('2. Геометрический смысл в сечении $y = 0$')
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-2.2, 2.2])
ax2.set_ylim([-2.2, 2.2])
ax2.legend(loc='lower right', fontsize=8.5)

plt.tight_layout()
plt.show()
