"""
traffic_visualization.py - animated four-way intersection (plotly).

Vehicles waiting in a queue are drawn as squares lined up behind the stop line;
vehicles that left during the last 3 s are drawn crossing the junction.  The four
traffic lights are coloured circles.  Press Play or drag the slider to animate.
"""
import numpy as np

from simulation.traffic_controller import (EW_GREEN, EW_YELLOW, NS_GREEN, NS_YELLOW)

try:
    import plotly.graph_objects as go
except ImportError:      # pragma: no cover
    go = None

GREEN, YELLOW, RED = "#22c55e", "#facc15", "#ef4444"
APP_COLORS = ["#2563eb", "#0d9488", "#ea580c", "#9333ea"]       # N, S, E, W
MAX_SHOW, SPACING, STOP = 14, 0.5, 1.2                          # vehicles drawn per queue, gap, stop-line distance
# unit vector (dx, dy) pointing AWAY from the junction, plus the lane offset, per approach
_DIRS = [((0, 1), (-0.5, 0)), ((0, -1), (0.5, 0)), ((1, 0), (0, 0.5)), ((-1, 0), (0, -0.5))]
_LIGHT_POS = [(-1.5, 1.6), (1.5, -1.6), (1.6, 1.5), (-1.6, -1.5)]
_LABEL_POS = [(-0.5, 8.6), (0.5, -8.6), (8.6, 0.5), (-8.6, -0.5)]


def signal_colors(state: int):
    """Colour of the N, S, E, W lights for a signal state code."""
    ns = GREEN if state == NS_GREEN else YELLOW if state == NS_YELLOW else RED
    ew = GREEN if state == EW_GREEN else YELLOW if state == EW_YELLOW else RED
    return [ns, ns, ew, ew]


def vehicle_points(queues, recent_dep):
    """Positions/colours of vehicles.  queues = 4 queue lengths; recent_dep = list of
    (age_s, departures[4]) for departures in the last seconds."""
    xs, ys, cs = [], [], []
    for j in range(4):
        (dx, dy), (ox, oy) = _DIRS[j]
        for i in range(min(int(queues[j]), MAX_SHOW)):
            d = STOP + SPACING * i
            xs.append(ox + dx * d); ys.append(oy + dy * d); cs.append(APP_COLORS[j])
        for age, dep in recent_dep:                       # vehicles crossing the junction
            if dep[j] > 0:
                d = 0.9 - 0.8 * age
                xs.append(ox + dx * d); ys.append(oy + dy * d); cs.append(APP_COLORS[j])
    return xs, ys, cs


def frame_traces(t, queue_ts, dep_ts, state_ts):
    recent = [(k, dep_ts[t - k]) for k in range(3) if t - k >= 0]
    xs, ys, cs = vehicle_points(queue_ts[t], recent)
    veh = go.Scatter(x=xs, y=ys, mode="markers", marker=dict(symbol="square", size=11, color=cs,
                     line=dict(width=1, color="white")), hoverinfo="skip", showlegend=False)
    sig = go.Scatter(x=[p[0] for p in _LIGHT_POS], y=[p[1] for p in _LIGHT_POS], mode="markers",
                     marker=dict(size=20, color=signal_colors(int(state_ts[t])),
                                 line=dict(width=2, color="#111827")), hoverinfo="skip", showlegend=False)
    lab = go.Scatter(x=[p[0] for p in _LABEL_POS], y=[p[1] for p in _LABEL_POS], mode="text",
                     text=[f"{n}: {int(q)}" for n, q in zip("NSEW", queue_ts[t])],
                     textfont=dict(size=14), hoverinfo="skip", showlegend=False)
    return [veh, sig, lab]


def build_intersection_figure(queue_ts, dep_ts, state_ts, t_start=0, n_seconds=120, step=2):
    """Animated figure covering [t_start, t_start + n_seconds)."""
    T = len(state_ts)
    t_start = int(min(max(0, t_start), T - 1))
    times = list(range(t_start, min(T, t_start + n_seconds), step))
    frames = [go.Frame(data=frame_traces(t, queue_ts, dep_ts, state_ts), name=str(t)) for t in times]
    fig = go.Figure(data=frame_traces(times[0], queue_ts, dep_ts, state_ts), frames=frames)
    road = "#9ca3af"
    fig.update_layout(
        shapes=[dict(type="rect", x0=-9.5, x1=9.5, y0=-1, y1=1, fillcolor=road, line_width=0, layer="below"),
                dict(type="rect", x0=-1, x1=1, y0=-9.5, y1=9.5, fillcolor=road, line_width=0, layer="below"),
                dict(type="line", x0=-9.5, x1=-1, y0=0, y1=0, line=dict(color="white", dash="dash", width=1), layer="below"),
                dict(type="line", x0=1, x1=9.5, y0=0, y1=0, line=dict(color="white", dash="dash", width=1), layer="below"),
                dict(type="line", x0=0, x1=0, y0=-9.5, y1=-1, line=dict(color="white", dash="dash", width=1), layer="below"),
                dict(type="line", x0=0, x1=0, y0=1, y1=9.5, line=dict(color="white", dash="dash", width=1), layer="below")],
        xaxis=dict(range=[-9.5, 9.5], visible=False, constrain="domain"),
        yaxis=dict(range=[-9.5, 9.5], visible=False, scaleanchor="x"),
        height=560, margin=dict(l=0, r=0, t=10, b=0), template="plotly_white",
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02, buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.15, len=0.82, currentvalue=dict(prefix="t = ", suffix=" s"),
                      steps=[dict(method="animate", label=str(t),
                                  args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))]) for t in times])])
    return fig


def density_level(avg_queue: float) -> str:
    """Traffic-density label used by the dashboard indicators."""
    return "Free flow" if avg_queue < 2 else "Moderate" if avg_queue < 6 else "Heavy" if avg_queue < 12 else "Congested"
