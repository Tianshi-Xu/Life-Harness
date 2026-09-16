# Prelaunch correction

The first airline launch stopped before any model request or scored episode.
The elevated process environment omitted the directory containing `uv`, so
the nested TauBench process could not start. The driver now prepends the
known `uv` directory explicitly. The experiment freeze was regenerated before
the first real evaluation.

The corrected launch then exposed an empty optional `MR_HOST` value from the
teacher environment, which had produced the invalid relative base URL
`/protocol/openai/v1`. It was stopped at 0/60 completed trajectories. The
launcher now applies the same public ModelRouter host default as `test_api.sh`;
partial, unscored output was discarded and the freeze was regenerated again.

The first complete ALFWorld identity baseline exposed one outer-driver-only
dependency: the cloned upstream module imports `python-dotenv`, which is absent
from the system Python used to coordinate native environments. A no-op
`load_dotenv` compatibility module is injected before importing the byte-for-byte
frozen upstream file. The completed baseline is unchanged and reused; no
candidate had been proposed or evaluated when this correction was made.

Airline is the only training arm with two trials per question. Its completed
identity score was 0.3667 over 60 trials, but the first proposer frontier
incorrectly treated each per-question mean as one binary trial and displayed
0.50. That proposer was stopped before writing a candidate. The adapter now
passes the original per-trial reward lists into upstream Meta-Harness. The
aborted proposer workspace is archived, and the completed 60-trial baseline is
reused without rerunning or altering any episode.
