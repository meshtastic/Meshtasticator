#!/usr/bin/env python3
"""Compare radio-policy variants on the same Meshtasticator scenario.

This is a small usability wrapper around loraMesh.py, not a second simulator.
Each policy run gets a fresh Config and calls the normal parse/run path, so the
comparison table stays aligned with the CLI users would run by hand.
"""

import argparse
import contextlib
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.config import Config  # noqa: E402
import loraMesh  # noqa: E402


POLICY_FLAGS = {
    "static": ("static CR with packet loss/capture physics", []),
    "dcr": ("Dynamic Coding Rate", ["--dcr"]),
    "dtp": ("Dynamic TX Power", ["--dtp"]),
}


@dataclass
class PolicySummary:
    name: str
    description: str
    messages: int
    sent: int
    received: int
    collisions: int
    phy_loss: int
    reach_percent: float | None
    useful_percent: float | None
    tx_air_percent: float | None
    cr_mix: str
    dtp_power_mix: str
    dtp_detected: float
    dtp_decodable: float
    output: str


@dataclass
class ThresholdFailure:
    policy: str
    metric: str
    delta_pp: float
    limit_pp: float
    message: str


def parse_policy_names(raw_policies):
    names = [name.strip() for name in raw_policies.split(",") if name.strip()]
    unknown = sorted(set(names) - set(POLICY_FLAGS))
    if unknown:
        known = ", ".join(POLICY_FLAGS)
        raise argparse.ArgumentTypeError(f"unknown policy {', '.join(unknown)}; choose from: {known}")
    if not names:
        raise argparse.ArgumentTypeError("at least one policy is required")
    return names


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="compare Meshtasticator radio policies on one scenario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python3 tools/radio_policy_compare.py
  python3 tools/radio_policy_compare.py --simtime-seconds 120 --period-seconds 5
  python3 tools/radio_policy_compare.py --policies static,dcr,dtp -- --no-clutter
""",
    )
    parser.add_argument("--preset", default="batumi", help="Packaged scenario preset to run")
    parser.add_argument("--simtime-seconds", type=positive_float, default=60.0, help="Simulation duration for every policy")
    parser.add_argument("--period-seconds", type=positive_float, default=5.0, help="Mean message-generation period for every policy")
    parser.add_argument(
        "--policies",
        type=parse_policy_names,
        default=parse_policy_names("static"),
        help="Comma-separated policies: static,dcr,dtp",
    )
    parser.add_argument(
        "--show-raw-output",
        action="store_true",
        help="Print each underlying loraMesh.py run before the comparison table",
    )
    parser.add_argument("--json-output", type=Path, help="Write a machine-readable CI report to this JSON file")
    parser.add_argument("--markdown-output", type=Path, help="Write a GitHub-friendly CI report to this Markdown file")
    parser.add_argument(
        "--max-reach-drop-pp",
        type=non_negative_float,
        help="Fail if any non-baseline policy loses more reach percentage points than this",
    )
    parser.add_argument(
        "--max-useful-drop-pp",
        type=non_negative_float,
        help="Fail if any non-baseline policy loses more useful-traffic percentage points than this",
    )
    parser.add_argument(
        "--max-tx-air-increase-pp",
        type=non_negative_float,
        help="Fail if any non-baseline policy spends more extra TX-air percentage points than this",
    )
    parser.add_argument(
        "lora_args",
        nargs=argparse.REMAINDER,
        help="Extra loraMesh.py arguments applied to every run; place them after --",
    )
    args = parser.parse_args(argv)
    if len(args.policies) < 2 and threshold_requested(args):
        parser.error("--max-* thresholds require at least two policies so there is a baseline and a candidate")
    return args


def threshold_requested(args):
    return (
        args.max_reach_drop_pp is not None
        or args.max_useful_drop_pp is not None
        or args.max_tx_air_increase_pp is not None
    )


def positive_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def non_negative_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def build_lora_args(args, policy_name):
    _, policy_flags = POLICY_FLAGS[policy_name]
    extra_args = list(args.lora_args)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]

    # The comparison intentionally enables packet-level loss and capture-aware
    # collisions for every policy. Without those two physics flags, CR and TX
    # power changes mostly affect airtime accounting, not delivery behavior.
    return [
        "--preset",
        args.preset,
        "--no-gui",
        "--simtime-seconds",
        str(args.simtime_seconds),
        "--period-seconds",
        str(args.period_seconds),
        "--phy-loss-model",
        "--capture-collision-model",
        *policy_flags,
        *extra_args,
    ]


def run_policy(policy_name, lora_args):
    conf = Config()
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        node_config = loraMesh.parse_params(conf, lora_args)
        results = loraMesh.run_simulation(conf, node_config)

    description, _ = POLICY_FLAGS[policy_name]
    return summarize_results(policy_name, description, results, stdout.getvalue())


def summarize_results(policy_name, description, results, output):
    return PolicySummary(
        name=policy_name,
        description=description,
        messages=int(results["messageSeq"]),
        sent=int(results["sent"]),
        received=int(results["nrReceived"]),
        collisions=int(results["nrCollisions"]),
        phy_loss=int(results["nrPhyLoss"]),
        reach_percent=as_percent(results["nodeReach"]),
        useful_percent=as_percent(results["usefulness"]),
        tx_air_percent=as_percent(results["txAirUtilizationRate"]),
        cr_mix=format_cr_mix(result_value(results, "dcrTxByCr", {})),
        dtp_power_mix=format_power_mix(result_value(results, "dtpTxByPower", {})),
        dtp_detected=float(result_value(results, "dtpMeanDetectedByTx", 0.0)),
        dtp_decodable=float(result_value(results, "dtpMeanSensedByTx", 0.0)),
        output=output,
    )


def result_value(results, key, default):
    try:
        return results[key]
    except KeyError:
        return default


def as_percent(value):
    numeric = float(value)
    if math.isnan(numeric):
        return None
    return numeric * 100.0


def format_percent(value):
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def format_cr_mix(cr_counts):
    total = sum(cr_counts.values())
    if not total:
        return "n/a"
    return "/".join(f"{100.0 * cr_counts.get(cr, 0) / total:.0f}" for cr in (5, 6, 7, 8))


def format_power_mix(power_counts):
    if not power_counts:
        return "n/a"
    ordered = sorted(power_counts.items(), reverse=True)
    return ",".join(f"{power}:{count}" for power, count in ordered)


def render_table(summaries):
    rows = [
        [
            "policy",
            "reach%",
            "useful%",
            "tx_air%",
            "msgs",
            "sent",
            "rx",
            "coll",
            "phy_loss",
            "cr5/6/7/8%",
            "power:tx",
            "cad/decodable",
        ]
    ]
    for summary in summaries:
        rows.append([
            summary.name,
            format_percent(summary.reach_percent),
            format_percent(summary.useful_percent),
            format_percent(summary.tx_air_percent),
            str(summary.messages),
            str(summary.sent),
            str(summary.received),
            str(summary.collisions),
            str(summary.phy_loss),
            summary.cr_mix,
            summary.dtp_power_mix,
            f"{summary.dtp_detected:.2f}/{summary.dtp_decodable:.2f}",
        ])

    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    lines = []
    for row_index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
        if row_index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def render_delta_table(summaries):
    if len(summaries) < 2:
        return ""

    baseline = summaries[0]
    lines = [f"\nDelta vs {baseline.name}:"]
    for summary in summaries[1:]:
        reach_delta = delta(summary.reach_percent, baseline.reach_percent)
        useful_delta = delta(summary.useful_percent, baseline.useful_percent)
        tx_air_delta = delta(summary.tx_air_percent, baseline.tx_air_percent)
        lines.append(
            f"  {summary.name}: "
            f"reach {reach_delta} pp, "
            f"useful {useful_delta} pp, "
            f"tx_air {tx_air_delta} pp, "
            f"sent {summary.sent - baseline.sent:+d}, "
            f"collisions {summary.collisions - baseline.collisions:+d}, "
            f"phy_loss {summary.phy_loss - baseline.phy_loss:+d}"
        )
    return "\n".join(lines)


def build_delta_rows(summaries):
    if len(summaries) < 2:
        return []

    baseline = summaries[0]
    rows = []
    for summary in summaries[1:]:
        rows.append({
            "baseline": baseline.name,
            "policy": summary.name,
            "reach_delta_pp": numeric_delta(summary.reach_percent, baseline.reach_percent),
            "useful_delta_pp": numeric_delta(summary.useful_percent, baseline.useful_percent),
            "tx_air_delta_pp": numeric_delta(summary.tx_air_percent, baseline.tx_air_percent),
            "sent_delta": summary.sent - baseline.sent,
            "collisions_delta": summary.collisions - baseline.collisions,
            "phy_loss_delta": summary.phy_loss - baseline.phy_loss,
        })
    return rows


def delta(value, baseline):
    if value is None or baseline is None:
        return "n/a"
    return f"{value - baseline:+.2f}"


def numeric_delta(value, baseline):
    if value is None or baseline is None:
        return None
    return value - baseline


def evaluate_thresholds(args, summaries):
    failures = []
    for row in build_delta_rows(summaries):
        policy = row["policy"]
        if args.max_reach_drop_pp is not None:
            failures.extend(check_min_delta(policy, "reach", row["reach_delta_pp"], -args.max_reach_drop_pp))
        if args.max_useful_drop_pp is not None:
            failures.extend(check_min_delta(policy, "useful", row["useful_delta_pp"], -args.max_useful_drop_pp))
        if args.max_tx_air_increase_pp is not None:
            failures.extend(check_max_delta(policy, "tx_air", row["tx_air_delta_pp"], args.max_tx_air_increase_pp))
    return failures


def check_min_delta(policy, metric, delta_pp, min_allowed_pp):
    if delta_pp is None or delta_pp >= min_allowed_pp:
        return []
    return [
        ThresholdFailure(
            policy=policy,
            metric=metric,
            delta_pp=delta_pp,
            limit_pp=min_allowed_pp,
            message=f"{policy} {metric} delta {delta_pp:+.2f} pp is below allowed {min_allowed_pp:+.2f} pp",
        )
    ]


def check_max_delta(policy, metric, delta_pp, max_allowed_pp):
    if delta_pp is None or delta_pp <= max_allowed_pp:
        return []
    return [
        ThresholdFailure(
            policy=policy,
            metric=metric,
            delta_pp=delta_pp,
            limit_pp=max_allowed_pp,
            message=f"{policy} {metric} delta {delta_pp:+.2f} pp is above allowed +{max_allowed_pp:.2f} pp",
        )
    ]


def summary_to_dict(summary):
    return {
        "policy": summary.name,
        "description": summary.description,
        "messages": summary.messages,
        "sent": summary.sent,
        "received": summary.received,
        "collisions": summary.collisions,
        "phy_loss": summary.phy_loss,
        "reach_percent": summary.reach_percent,
        "useful_percent": summary.useful_percent,
        "tx_air_percent": summary.tx_air_percent,
        "cr_mix": summary.cr_mix,
        "dtp_power_mix": summary.dtp_power_mix,
        "dtp_mean_cad_detected_receivers": summary.dtp_detected,
        "dtp_mean_decodable_receivers": summary.dtp_decodable,
    }


def build_report(args, summaries, failures):
    extra_args = list(args.lora_args)
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]

    return {
        "scenario": {
            "preset": args.preset,
            "simtime_seconds": args.simtime_seconds,
            "period_seconds": args.period_seconds,
            "policies": args.policies,
            "extra_lora_args": extra_args,
            "physics_flags": ["--phy-loss-model", "--capture-collision-model"],
        },
        "summaries": [summary_to_dict(summary) for summary in summaries],
        "deltas": build_delta_rows(summaries),
        "thresholds": {
            "max_reach_drop_pp": args.max_reach_drop_pp,
            "max_useful_drop_pp": args.max_useful_drop_pp,
            "max_tx_air_increase_pp": args.max_tx_air_increase_pp,
        },
        "failures": [failure.__dict__ for failure in failures],
    }


def write_json_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report):
    lines = [
        "# Meshtasticator Radio Policy Comparison",
        "",
        f"- preset: `{report['scenario']['preset']}`",
        f"- simtime: `{report['scenario']['simtime_seconds']}` seconds",
        f"- period: `{report['scenario']['period_seconds']}` seconds",
        f"- policies: `{', '.join(report['scenario']['policies'])}`",
        "",
        "| policy | reach% | useful% | tx_air% | msgs | sent | rx | coll | phy_loss | CR5/6/7/8% | power:tx | CAD/decodable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for summary in report["summaries"]:
        lines.append(
            "| {policy} | {reach} | {useful} | {tx_air} | {messages} | {sent} | {received} | "
            "{collisions} | {phy_loss} | {cr_mix} | {dtp_power_mix} | {detected:.2f}/{decodable:.2f} |".format(
                policy=summary["policy"],
                reach=format_percent(summary["reach_percent"]),
                useful=format_percent(summary["useful_percent"]),
                tx_air=format_percent(summary["tx_air_percent"]),
                messages=summary["messages"],
                sent=summary["sent"],
                received=summary["received"],
                collisions=summary["collisions"],
                phy_loss=summary["phy_loss"],
                cr_mix=summary["cr_mix"],
                dtp_power_mix=summary["dtp_power_mix"],
                detected=summary["dtp_mean_cad_detected_receivers"],
                decodable=summary["dtp_mean_decodable_receivers"],
            )
        )

    if report["deltas"]:
        lines.extend(["", "## Delta vs Baseline", ""])
        lines.append("| policy | reach pp | useful pp | tx_air pp | sent | collisions | phy_loss |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in report["deltas"]:
            lines.append(
                "| {policy} | {reach} | {useful} | {tx_air} | {sent:+d} | {collisions:+d} | {phy_loss:+d} |".format(
                    policy=row["policy"],
                    reach=format_delta_value(row["reach_delta_pp"]),
                    useful=format_delta_value(row["useful_delta_pp"]),
                    tx_air=format_delta_value(row["tx_air_delta_pp"]),
                    sent=row["sent_delta"],
                    collisions=row["collisions_delta"],
                    phy_loss=row["phy_loss_delta"],
                )
            )

    if report["failures"]:
        lines.extend(["", "## Threshold Failures", ""])
        for failure in report["failures"]:
            lines.append(f"- {failure['message']}")
    else:
        lines.extend(["", "No threshold failures."])

    return "\n".join(lines) + "\n"


def format_delta_value(value):
    if value is None:
        return "n/a"
    return f"{value:+.2f}"


def main(argv=None):
    args = parse_args(argv)
    loraMesh.configure_logging()

    summaries = []
    for policy_name in args.policies:
        lora_args = build_lora_args(args, policy_name)
        print(f"Running {policy_name}: loraMesh.py {' '.join(lora_args)}", file=sys.stderr)
        summary = run_policy(policy_name, lora_args)
        if args.show_raw_output:
            print(f"\n===== raw output: {policy_name} =====")
            print(summary.output.rstrip())
        summaries.append(summary)

    failures = evaluate_thresholds(args, summaries)
    report = build_report(args, summaries, failures)
    if args.json_output:
        write_json_report(args.json_output, report)
    if args.markdown_output:
        write_markdown_report(args.markdown_output, report)

    print("\nRadio policy comparison")
    print(render_table(summaries))
    delta_table = render_delta_table(summaries)
    if delta_table:
        print(delta_table)
    if all(summary.messages == 0 for summary in summaries):
        print(
            "\nNo messages were generated; increase --simtime-seconds or lower --period-seconds for a useful comparison."
        )
    if failures:
        print("\nThreshold failures:")
        for failure in failures:
            print(f"  - {failure.message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
