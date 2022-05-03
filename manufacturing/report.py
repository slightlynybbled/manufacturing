from datetime import datetime
import logging
from pathlib import Path
from os import chdir, getcwd
from os.path import splitext
from subprocess import check_call

import matplotlib.pyplot as plt
import pandas as pd

from manufacturing.analysis import normality_test, suggest_specification_limits
from manufacturing.data_import import parse_col_for_limits
from manufacturing.visual import ppk_plot, cpk_plot, control_chart

_logger = logging.getLogger(__name__)


def generate_production_report(
    input_file: (str, Path),
    output_file: Path = None,
    title: str = "Production Report",
    **kwargs,
):
    """


    :param input_file: the input file to be analyzed
    :param output_file: the output file to be analyzed
    :param title: the title of the report
    :param kwargs: the keyword args to be passed into `pandas.read_csv()` or `pandas.read_excel()` methods
    :return:
    """
    _logger.info("attempting to generate report...")

    if not isinstance(input_file, Path):
        _logger.debug("coercing `str` to `Path`")
        input_file = Path(input_file)

    if input_file.name.endswith("csv"):
        _logger.debug("csv file detected")
        df = pd.read_csv(input_file, **kwargs)
    elif input_file.name.endswith("csv"):
        _logger.debug("ms excel file detected")
        df = pd.read_excel(input_file, **kwargs)
    else:
        raise ValueError("the input file extension is not currently supported")

    build_path = Path("./build")
    build_path.mkdir(parents=True, exist_ok=True)

    text = f"# {title}\n\n"
    text += f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n'

    for c in df.columns:
        _logger.info(f'analyzing column "{c}"...')
        lcl, ucl = parse_col_for_limits(c)

        text += f"## Column: {c}\n\n"

        normal = normality_test(data=df[c])
        if not normal:
            text += (
                "Normality test indicates that the data is likely "
                "not normally distributed, meaning that the suggested "
                "limits are not based on a normally "
                "distributed parameter.\n\n"
            )

        if lcl is not None and ucl is not None:
            text += (
                f"Established limits:\n\n * LCL = {lcl:.02g}\n * UCL = {ucl:.02g}\n\n"
            )
        else:
            lcl, ucl = suggest_specification_limits(df[c])
            text += (
                f"Recommended limits:\n\n * LCL = {lcl:.02g}\n * UCL = {ucl:.02g}\n\n"
            )

        fig_name = c.split("(")[0].strip()

        # generate visual Ppk plot
        fig, ax = plt.subplots(1, 1)
        ppk_plot(
            df[c], upper_specification_limit=ucl, lower_specification_limit=lcl, ax=ax
        )
        plot_name = build_path / f"ppk_plot_{fig_name}.png"
        fig.savefig(plot_name)
        text += f"![Ppk Plot: {fig_name}]({plot_name.name})\n\n"

        # generate zpk subgroup plot
        fig, axs = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [4, 1]})
        cpk_plot(
            df[c], upper_specification_limit=ucl, lower_specification_limit=lcl, axs=axs
        )
        plot_name = build_path / f"cpk_plot_{fig_name}.png"
        fig.savefig(plot_name)
        text += f"![Cpk Plot: {fig_name}]({plot_name.name})\n\n"

        # generate zone control plot
        fig = control_chart(df[c])
        plot_name = build_path / f"zone_control_chart_{fig_name}.png"
        fig.savefig(plot_name)
        text += f"![Zone Control Plot: {fig_name}]({plot_name.name})\n\n"

        _logger.info(f'analysis of column "{c}" complete!')

    with (build_path / "report.md").open("w") as f:
        f.write(text)

    if output_file is None:
        return

    cwd = getcwd()
    chdir(build_path.absolute())

    extension = splitext(str(output_file))[1].replace(".", "")
    if extension == "pdf":
        args = ["pandoc", f"report.md", "-o", f"{title}.pdf"]
    elif extension == "html":
        args = ["pandoc", "-s", "report.md", "-o", f"{title}.html"]
    else:
        raise ValueError(f'extension "{extension}" not supported')
    _logger.info(f"executing {args}")
    check_call(args)
    chdir(cwd)

    _logger.info(
        f"report generation complete; artifact may be found at {build_path.absolute()}"
    )
