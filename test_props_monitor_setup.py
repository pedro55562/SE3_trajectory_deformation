import uaibot as ub

from props import create_monitor_prop, create_computer_setup


objects = []

# Monitor isolado: facilita verificar contato suporte-frame
objects += create_monitor_prop(
    htm=ub.Utils.trn([-2.0, 0.0, 0.0]),
    name="monitor_isolated",
)

# Setup simples
objects += create_computer_setup(
    htm=ub.Utils.trn([0.0, 0.0, 0.0]),
    name="setup_single",
    dual_monitor=False,
    include_speakers=True,
    include_webcam=False,
    include_tower=True,
    tower_side="right",
)

# Setup dual
objects += create_computer_setup(
    htm=ub.Utils.trn([2.8, 0.0, 0.0]),
    name="setup_dual",
    dual_monitor=True,
    include_speakers=True,
    include_webcam=False,
    include_tower=True,
    tower_side="left",
)

sim = ub.Simulation.create_sim_grid(objects)
sim.run()
