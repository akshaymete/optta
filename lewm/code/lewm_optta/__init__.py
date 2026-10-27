"""OpTTA on LeWM: AdaJEPA test-time adaptation plus the optimism term, for LeWM.

A thin, explicit layer over the installed `stable_worldmodel` stack, with no patches to it.
Entry point: `python -m lewm_optta.eval task=<task> arm=<frozen|adajepa|optta> cell=<cell> seed=<seed>`.
"""

__version__ = "0.1.0"
