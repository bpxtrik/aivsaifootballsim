import math
from typing import Tuple


def clamp(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def length(x: float, y: float) -> float:
    return math.hypot(x, y)


def normalize(x: float, y: float) -> Tuple[float, float]:
    l = length(x, y)
    if l == 0:
        return 0.0, 0.0
    return x / l, y / l


def dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def resolve_circle_circle_collision(
    x1: float,
    y1: float,
    vx1: float,
    vy1: float,
    r1: float,
    m1: float,
    x2: float,
    y2: float,
    vx2: float,
    vy2: float,
    r2: float,
    m2: float,
    restitution: float = 0.8,
):
    dx = x2 - x1
    dy = y2 - y1
    dist = length(dx, dy)
    min_dist = r1 + r2

    if dist == 0:
        # Prevent division by zero; nudge apart on x axis
        nx, ny = 1.0, 0.0
        dist = min_dist
    else:
        nx, ny = dx / dist, dy / dist

    overlap = min_dist - dist
    if overlap > 0:
        # Positional correction: push objects apart proportionally to mass (heavier moves less)
        total_mass = m1 + m2
        if total_mass == 0:
            total_mass = 1.0
        correction_x = nx * overlap
        correction_y = ny * overlap
        x1 -= correction_x * (m2 / total_mass)
        y1 -= correction_y * (m2 / total_mass)
        x2 += correction_x * (m1 / total_mass)
        y2 += correction_y * (m1 / total_mass)

        # Relative velocity
        rvx = vx2 - vx1
        rvy = vy2 - vy1
        vel_along_normal = dot(rvx, rvy, nx, ny)

        if vel_along_normal < 0:
            # Compute impulse scalar
            j = -(1 + restitution) * vel_along_normal
            j /= (1 / (m1 if m1 > 0 else 1)) + (1 / (m2 if m2 > 0 else 1))

            impulse_x = j * nx
            impulse_y = j * ny

            vx1 -= impulse_x / (m1 if m1 > 0 else 1)
            vy1 -= impulse_y / (m1 if m1 > 0 else 1)
            vx2 += impulse_x / (m2 if m2 > 0 else 1)
            vy2 += impulse_y / (m2 if m2 > 0 else 1)

    return x1, y1, vx1, vy1, x2, y2, vx2, vy2


def resolve_circle_wall_collision(
    x: float,
    y: float,
    vx: float,
    vy: float,
    r: float,
    width: int,
    height: int,
    restitution: float = 0.8,
):
    if x - r < 0:
        x = r
        vx = -vx * restitution
    elif x + r > width:
        x = width - r
        vx = -vx * restitution

    if y - r < 0:
        y = r
        vy = -vy * restitution
    elif y + r > height:
        y = height - r
        vy = -vy * restitution

    return x, y, vx, vy
