"""Is gnubg 3-ply a materially better teacher than gnubg 2-ply? (runs INSIDE gnubg)

Experiment 24 rejected a deeper teacher on two grounds — rollouts cost ~0.3 s per
trial per position, and `gnubg-cli` prints "Rollout done." with nothing parseable
after it. Both objections were about *rollouts via the CLI*. The embedded Python
API is a different door: it honours an arbitrary ply and returns the probability
vector as data. Measured on box B, per position:

    0-ply  13 ms   1-ply  16 ms   2-ply  31 ms   3-ply  324 ms   4-ply  6038 ms

So 3-ply costs ~10x a 2-ply label, not the ~10^5x that killed rollouts, and 4-ply
is out of reach. That makes 3-ply affordable — but affordable is not the same as
worth it, and relabelling millions of positions is a long run to start on a hunch.

This measures the thing that decides it: how far apart the two teachers' labels
actually are. If 3-ply says what 2-ply says, distilling it changes nothing, and we
stop before building anything. That is the §24 lesson applied a third time — price
the teacher before designing around it.

Reports mean and tail |Δ| in equity and in win probability, plus the share of
positions where the two disagree by more than a plausible training noise floor.

    set BGNN_POS_IN=<file of position ids, one per line>
    set BGNN_CSV_OUT=<csv to write>
    gnubg-cli.exe -q -t -p trainer/teacher_depth.py
"""

import os
import time

import gnubg

POS_IN = os.environ.get("BGNN_POS_IN")
CSV_OUT = os.environ.get("BGNN_CSV_OUT")
DEEP = int(os.environ.get("BGNN_DEEP_PLY", "3"))
BASE = int(os.environ.get("BGNN_BASE_PLY", "2"))


def ctx(plies):
    gnubg.command(f"set evaluation chequerplay eval plies {plies}")
    return gnubg.evalcontext()


def main():
    if not POS_IN or not CSV_OUT:
        raise SystemExit("set BGNN_POS_IN and BGNN_CSV_OUT")
    with open(POS_IN) as f:
        ids = [ln.strip() for ln in f if ln.strip()]

    gnubg.command("set player 0 human")
    gnubg.command("set player 1 human")
    # Every `set board` otherwise echoes the whole ASCII board, which dwarfs the
    # useful output and costs real time over thousands of positions.
    gnubg.command("set display off")
    gnubg.command("new session")
    gnubg.command("new game")

    # Written and flushed per row. 3-ply on a real contact position costs seconds,
    # not the 0.3s the opening position suggested, so any budget will cut this run
    # off part-way — and a run that buffers everything to the end yields nothing
    # when it is killed.
    n = 0
    t0 = time.time()
    with open(CSV_OUT, "w") as f:
        f.write("pos_id,win_base,eq_base,win_deep,eq_deep\n")
        f.flush()
        for i, pid in enumerate(ids):
            gnubg.command(f"set board {pid}")
            gnubg.command("set turn 0")
            try:
                board, cube = gnubg.board(), gnubg.cubeinfo()
                base = gnubg.evaluate(board, cube, ctx(BASE))
                deep = gnubg.evaluate(board, cube, ctx(DEEP))
            except Exception:
                continue      # gnubg declines a position; drop it, never reuse
            # (win, win_g, win_bg, lose_g, lose_bg, equity)
            f.write(f"{pid},{base[0]:.6f},{base[5]:.6f},{deep[0]:.6f},{deep[5]:.6f}\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                print(f"  {n}/{len(ids)}  {(time.time()-t0)/n:.2f}s/pos", flush=True)
    print(f"wrote {CSV_OUT}: {n} of {len(ids)} positions at {BASE}-ply vs "
          f"{DEEP}-ply in {time.time()-t0:.0f}s ({(time.time()-t0)/max(n,1):.2f}s/pos)")


main()
