import importlib.metadata

from manufacturing.analysis import calc_ppk, suggest_specification_limits
from manufacturing.data_import import import_csv, import_excel
from manufacturing.report import generate_production_report
from manufacturing.visual import (
    control_chart,
    control_chart_base,
    control_plot,
    cpk_plot,
    ppk_plot,
    x_mr_chart,
    xbar_r_chart,
    xbar_s_chart,
)


__all__ = [
    "calc_ppk",
    "control_chart",
    "control_chart_base",
    "control_plot",
    "cpk_plot",
    "generate_production_report",
    "import_csv",
    "import_excel",
    "ppk_plot",
    "suggest_specification_limits",
    "x_mr_chart",
    "xbar_r_chart",
    "xbar_s_chart",
]

__version__ = importlib.metadata.version("manufacturing")
