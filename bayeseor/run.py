import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pymultinest.solve import solve

from .posterior import PowerSpectrumPosteriorProbability, PriorC
from .utils import mpiprint


def run(
    *,
    pspp: PowerSpectrumPosteriorProbability,
    priors: Sequence[float],
    n_live_points: int | None = None,
    calc_avg_eval: bool = False,
    avg_iters: int = 10,
    out_dir: Path | str = "./",
    sampler: str = "multinest",
    rank: int = 0,
):
    """
    Run a power spectrum analysis.

    Parameters
    ----------
    pspp : :class:`bayeseor.posterior.PowerSpectrumPosteriorProbability`
        Power spectrum posterior class.
    priors : list of float
        Prior [min, max] for each k bin as a list, e.g. [[min1, max1],
        [min2, max2], ...].
    n_live_points : int, optional
        Number of live points. Defaults to None (sets the number of live
        points to `25 * pspp.k_vals.size`).
    calc_avg_eval : bool, optional
        Estimate the average evaluation time by calculating the posterior
        probability of the mean value of each prior range `avg_iters` times.
        Only computed if `rank` is 0. Defaults to False.
    avg_iters : int, optional
        Number of iterations to use to calculate the average posterior
        probability evaluation time. Used only if `calc_avg_eval` is True.
        Defaults to 10.
    out_dir : pathlib.Path or str, optional
        Sampler output directory. Must be less than 100 characters for
        compatibility with Multinest (the fortran code only supports a string
        with character length <= 100). It is encouraged you run BayesEoR from
        the directory where you would like the outputs to be written for this
        reason. Defaults to "./".
    sampler : {'multinest', 'polychord'}, optional
        Case insensitive string specifying the sampler, one of 'multinest' or
        'polychord'. Defaults to 'multinest'.
    rank : int, optional
        MPI rank. Defaults to 0.

    """
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler_output_base = (out_dir / "data-").as_posix()

    sampler = sampler.lower()
    if sampler == "multinest":
        # The longest file name created by MultiNest is:
        # <file-root-dir><file-root>post_equal_weights.dat
        # which is 22 characters longer than <file-root-dir><file-root>.
        # Assuming the file-root is "data-", which is currently hard-coded above,
        # this means the maximum allowed out_dir path length is
        # max_path_length = 100 - 22 - 5 = 73 characters.
        # Check that the output directory path length is <= max_path_length characters
        # ---
        # Possible upgrade (JB): check MultiNest for the output file names
        # to calculate max_path_length directly from pymultinest.
        # ---
        max_path_length = 73
        assert len(out_dir.as_posix()) <= max_path_length, (
            "When using MultiNest, the full file path must be <= 100 characters in \n"
            "length.\n Using the default `data-` file prefix, the path to the sampler\n"
            f"directory output `out_dir` must be <= {max_path_length} characters in \n"
            "length. Either <file-root-dir> must be shortened or the MultiNest file \n"
            "prefix must be changed from `data-` to a shorter string."
        )

        # Log-likelihood wrapper function for MultiNest
        def loglikelihood(theta, calc_likelihood=pspp.posterior_probability):
            return calc_likelihood(theta)[0]
    elif sampler == "polychord":
        raise NotImplementedError("PolyChord will be supported in the future.")
    else:
        raise ValueError("sampler must be one of 'multinest' or 'polychord'")

    nkbins = pspp.k_vals.size
    if n_live_points is None:
        n_live_points = 25 * nkbins
    prior_c = PriorC(priors)
    # Suppress verbose output for power spectrum analysis only
    pspp.verbose = False

    if calc_avg_eval:
        # Compute the average posterior calculation time for
        # reference and check that this calculation returns
        # a finite value for the posterior probability
        mpiprint(
            "\nCalculating average posterior probability evaulation time:",
            style="bold",
            rank=rank,
        )
        start = time.time()
        pspp_verbose = pspp.verbose
        pspp.verbose = False
        for _ in range(avg_iters):
            post = pspp.posterior_probability(np.array(priors).mean(axis=1))[0]
            if not np.isfinite(post):
                # rank kwarg deliberately omitted to print warning on all ranks
                mpiprint(
                    f"{rank}: WARNING: Infinite value returned in posterior calculation!",
                    style="bold red",
                    # rank=rank,
                )
        # rank kwarg deliberately omitted to print evaluation time on all ranks
        avg_eval_time = (time.time() - start) / avg_iters
        mpiprint(
            f"{rank}: Average evaluation time: {avg_eval_time} s",
            # rank=rank,
            end="\n\n",
        )

    if sampler == "multinest":
        result = solve(
            LogLikelihood=loglikelihood,
            Prior=prior_c.prior_func,
            n_dims=nkbins,
            outputfiles_basename=sampler_output_base,
            n_live_points=n_live_points,
        )
    mpiprint("\nSampling complete!", rank=rank, end="\n\n")
