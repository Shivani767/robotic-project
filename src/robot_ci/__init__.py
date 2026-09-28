"""ROBOT CI - automated behavioural regression testing for physical AI.

The package is layered so that each layer depends only on the ones below it:

    models           versioned, frozen data schemas
    core             errors, seeding, hashing, I/O, workspace
    simulation       SimulatorBackend ABC + mock/synthetic/mujoco backends
    scenarios        YAML loading, catalogue, grid expansion
    policies         RobotPolicy interface and implementations
    tasks            task definitions and success criteria
    sensors          corruption engine
    perception       RGB/depth -> object estimates
    sensor_fusion    probabilistic fusion with covariance
    world_model      believed world state with snapshots
    metrics          measurement suite
    experiments      ExperimentRunner and artifact writing
    regression       conditional regression detection
    counterexamples  counterexample search and minimization
    boundary         behavioural boundary mapping
    diagnosis        failure taxonomy, ablation, diagnosis
    statistics       intervals, tests, effect sizes
    database         SQLite experiment store
    reporting        HTML/JSON reports
    cli              Typer command-line interface

Only the top-level name and version are imported eagerly, so ``import
robot_ci`` stays cheap.
"""

from robot_ci.core.version import ENVIRONMENT_VERSION, SCHEMA_VERSION, SIMULATOR_VERSION, VERSION

__all__ = ["ENVIRONMENT_VERSION", "SCHEMA_VERSION", "SIMULATOR_VERSION", "VERSION", "__version__"]

__version__ = VERSION
