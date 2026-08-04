"""Transcribe gnubg's loaded match equity table into ``trainer/met_kazaross.py``.

This runs *inside* gnubg, not under our venv — gnubg embeds its own Python and
``import gnubg`` only resolves there. Run it with::

    set BGNN_MET_OUT=<repo>\trainer\met_kazaross.py
    gnubg-cli.exe -q -t -p <absolute path to this file>

Both paths must be absolute. gnubg resolves ``-p`` against its own directory
rather than the shell's cwd (it reports "Python file not found" otherwise), and
it execs the script without setting ``__file__``, so the destination cannot be
derived from this file's location — hence the environment variable.

``gnubg.met()`` returns ``[pre, post_player0, post_player1]``, where ``pre`` is a
64x64 grid of match-winning chances and the two post-Crawford lists are 64 long.
Both post lists are identical for a symmetric table; we keep one. We truncate to
25-away because that is the extent of the real Kazaross data (gnubg projects
beyond it), and because a match longer than 25 points does not occur.

Whatever table gnubg has loaded is what gets written, so check the name it prints
before committing the result — the default is Kazaross XG2.
"""

import json
import os

import gnubg

OUT = os.environ.get("BGNN_MET_OUT")
N = 25


def main() -> None:
    if not OUT:
        raise SystemExit("set BGNN_MET_OUT to the destination .py path first")
    gnubg.command("show matchequity")  # prints the table's name and provenance
    pre_raw, post_raw, _ = gnubg.met()
    pre = [[round(pre_raw[a][b], 6) for b in range(N)] for a in range(N)]
    post = [round(post_raw[i], 6) for i in range(N)]

    # A published MET is antisymmetric by construction; if this trips, the table
    # was read wrongly rather than being merely inaccurate.
    for a in range(N):
        for b in range(N):
            assert abs(pre[a][b] + pre[b][a] - 1.0) < 1e-4, (a, b)

    body = ["PRE: list[list[float]] = ["]
    for row in pre:
        body.append("    [" + ", ".join(f"{v:.6f}" for v in row) + "],")
    body.append("]")
    body.append("")
    body.append("POST: list[float] = [")
    for i in range(0, N, 5):
        body.append("    " + ", ".join(f"{v:.6f}" for v in post[i:i + 5]) + ",")
    body.append("]")

    # Explicit utf-8: gnubg's embedded interpreter defaults to the Windows ANSI
    # codepage, which writes the docstring's dashes as bytes Python then refuses
    # to import.
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(HEADER.format(n=N))
        f.write("\n".join(body) + "\n")
    print(f"wrote {OUT}: {N}x{N} pre-Crawford + {N} post-Crawford")


HEADER = '''"""Kazaross XG2 match equity table — transcribed from GNU Backgammon.

    Copyright (C) 2011 Neil Kazaross

    Table rolled up to 9 point match by eXtreme Gammon. Then uses R/K MET
    which was rolled up to 15 and extrapolated to 25 points.

    Transcribed for use by GNUbg by Michael Petch <mpetch@capp-sysware.com>

    This file is distributed as a part of the GNU Backgammon program.

    Copying and distribution of this file, with or without modification,
    are permitted in any medium without royalty provided the copyright
    notice and this notice are preserved.  This file is offered as-is,
    without any warranty.

The notice above is reproduced from gnubg's ``met/Kazaross-XG2.xml``, whose
numbers these are. It is the GNU all-permissive licence, not the GPL: copying,
modification and redistribution are allowed royalty-free in any medium, and the
single condition is that the notice travels with the data. So it must survive
transcription into this file, regeneration of this file, and packaging into the
distributed exe — do not strip it.

Generated data, do not hand-edit. Regenerate with ``trainer/dump_met.py``.

Kazaross XG2 is a rollout-derived table (XG rollouts to 9 points, GNUbg Supremo
full rollouts to 15, take points projected to 25) and is gnubg's own default. It
is *cubeful*: unlike a cubeless recursion it prices the trailer's ability to
double their way back into the match, which is where the two disagree most.

``PRE[a-1][b-1]`` is the match-winning chance of the player who needs ``a`` more
points against an opponent needing ``b``, with the Crawford game still to come
(so entries with ``b == 1`` are Crawford-game values — played with no cube).

``POST[n-1]`` is the MWC of the player needing ``n`` once the Crawford game has
been played and the opponent needs 1. It is a separate table because the cube
comes back: post-Crawford at 2-away/1-away the trailer is on 48.8%, not the
32.3% the Crawford game is worth — the largest single correction in the table.
"""

MAX_AWAY = {n}

'''


if __name__ == "__main__":
    main()
