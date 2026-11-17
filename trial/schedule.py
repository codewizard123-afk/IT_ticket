"""Seeded crossover schedule for the timed MTTR trial (architecture §9.3).

Group A works T01–T06 manually and then T07–T12 assisted; group B does the reverse (T01–T06
assisted, then T07–T12 manual). Participants are assigned to A/B alternately after a seeded
shuffle, and the order inside each block is shuffled per participant with the same seed, so the
schedule is reproducible. Everyone starts with one practice ticket (not timed in the analysis).

Usage::

    python -m trial.schedule alice bob carol dave            # print the schedule
    python -m trial.schedule alice bob carol dave --csv trial/results/schedule.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ui.settings import load_ui_config

FIELDS = ("participant", "group", "seq", "ticket_id", "mode", "practice")


@dataclass(frozen=True)
class Slot:
    participant: str
    group: str
    seq: int
    ticket_id: str
    mode: str
    practice: bool = False


def assign_groups(participants: list[str], seed: int) -> dict[str, str]:
    """Balanced A/B assignment: shuffle with the seed, then alternate A, B, A, B…"""
    if len(set(participants)) != len(participants):
        raise ValueError("participant names must be unique")
    order = sorted(participants)
    random.Random(seed).shuffle(order)
    return {p: "AB"[i % 2] for i, p in enumerate(order)}


def make_schedule(
    participants: list[str],
    seed: int | None = None,
    first_block: list[str] | None = None,
    second_block: list[str] | None = None,
    practice_ticket: str | None = None,
) -> list[Slot]:
    """Every participant: 1 practice ticket, then 6 tickets in one mode and 6 in the other."""
    cfg = load_ui_config()["trial"]
    seed = cfg["seed"] if seed is None else seed
    first_block = list(first_block or cfg["manual_block"])
    second_block = list(second_block or cfg["assisted_block"])
    practice_ticket = practice_ticket or cfg["practice_ticket"]
    if set(first_block) & set(second_block):
        raise ValueError("the two blocks must not share tickets")
    groups = assign_groups(participants, seed)
    slots: list[Slot] = []
    for p in sorted(participants):
        g = groups[p]
        rng = random.Random(f"{seed}:{p}")
        a, b = first_block[:], second_block[:]
        rng.shuffle(a)
        rng.shuffle(b)
        # A: first block manual, second assisted. B: first block assisted, second manual.
        modes = ("manual", "assisted") if g == "A" else ("assisted", "manual")
        # practice: one demo ticket on the Analyze page, so everyone has seen the UI once
        slots.append(Slot(p, g, 0, practice_ticket, "assisted", practice=True))
        seq = 1
        for block, mode in ((a, modes[0]), (b, modes[1])):
            for ticket in block:
                slots.append(Slot(p, g, seq, ticket, mode))
                seq += 1
    return slots


def write_csv(slots: list[Slot], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(asdict(s) for s in slots)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seeded crossover schedule for the MTTR trial")
    parser.add_argument("participants", nargs="+", help="participant names (>= 4 recommended)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--csv", type=Path, default=None, help="write the schedule to this CSV")
    args = parser.parse_args(argv)
    if len(args.participants) < 4:
        print("warning: the protocol asks for at least 4 participants", file=sys.stderr)
    slots = make_schedule(args.participants, seed=args.seed)
    if args.csv:
        write_csv(slots, args.csv)
        print(f"wrote {len(slots)} slots to {args.csv}")
    current = None
    for s in slots:
        if s.participant != current:
            current = s.participant
            print(f"\n{s.participant} (group {s.group})")
        tag = "practice" if s.practice else f"{s.seq:2d}"
        print(f"  {tag:>8}  {s.ticket_id}  {s.mode}")


if __name__ == "__main__":
    main()
