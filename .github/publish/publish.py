#!/usr/bin/env python3
"""Publish a build run's container artifacts to ECR (OCI) and S3 (SIF).

Runs only after :mod:`authorize` has authorized the publish and produced the
selector. Splits cleanly into:

* :func:`plan` — pure logic: map each downloaded artifact leg to its ECR tag
  and S3 URI. Unit-tested, no I/O.
* :func:`main` — thin glue: ECR login, then run each planned copy via skopeo /
  aws with explicit argument lists (no shell, so no quoting pitfalls).

A native build produces one leg per arch; a release build one leg per
microarchitecture target. Each leg is a pair of downloaded artifacts:
``<image>-oci/<image>.tar`` and ``<image>-sif/<image>.sif`` where ``<image>`` is
``palace-<arch_label>-<shorthash>``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublishItem:
    image_name: str
    arch_label: str
    oci_tar: Path
    sif: Path
    ecr_tag: str  # full ECR ref, e.g. <registry>/palace:main-sapphirerapids
    s3_uri: str  # full s3:// destination for the SIF


def arch_label_from_image(image_name: str) -> str:
    """``palace-<arch_label>-<shorthash>`` -> ``<arch_label>``.

    Arch labels are ``[a-z0-9_]`` and the short hash is hex, so stripping the
    leading ``palace-`` and the trailing ``-<hash>`` is unambiguous.
    """
    if not image_name.startswith("palace-"):
        raise ValueError(f"image name '{image_name}' does not start with 'palace-'")
    rest = image_name[len("palace-"):]
    label, _, short_hash = rest.rpartition("-")
    if not label or not short_hash:
        raise ValueError(f"cannot parse arch label from image name '{image_name}'")
    return label


def s3_prefix_for(selector: str) -> str:
    """S3 key prefix for a selector, preserving the dev/<branch> layout.

    ``dev-<branch>`` selectors publish under ``dev/<branch>/``; ``main`` and
    release selectors publish under ``<selector>/``.
    """
    if selector.startswith("dev-"):
        return f"dev/{selector[len('dev-'):]}"
    return selector


def plan(artifacts_dir: Path, selector: str, registry: str, ecr_repo: str, s3_bucket: str) -> list[PublishItem]:
    """Build the publish plan from the downloaded artifacts. Pure; no I/O beyond
    listing the artifact directory."""
    items: list[PublishItem] = []
    for oci_tar in sorted(artifacts_dir.glob("*-oci/*.tar")):
        image_name = oci_tar.parent.name[: -len("-oci")]  # strip "-oci" dir suffix
        arch_label = arch_label_from_image(image_name)
        sif = artifacts_dir / f"{image_name}-sif" / f"{image_name}.sif"
        ecr_tag = f"{registry}/{ecr_repo}:{selector}-{arch_label}"
        s3_uri = f"s3://{s3_bucket}/{s3_prefix_for(selector)}/{arch_label}.sif"
        items.append(PublishItem(image_name, arch_label, oci_tar, sif, ecr_tag, s3_uri))
    return items


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--artifacts-dir", default="./artifacts")
    args = parser.parse_args(argv)

    region = os.environ["AWS_REGION"]
    ecr_repo = os.environ["ECR_REPOSITORY"]
    s3_bucket = os.environ["S3_CONTAINERS_BUCKET"]

    account_id = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"::add-mask::{account_id}")
    registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"

    items = plan(Path(args.artifacts_dir), args.selector, registry, ecr_repo, s3_bucket)
    if not items:
        print("::error::no OCI artifacts found to publish", file=sys.stderr)
        return 1

    # Log skopeo into ECR once (writes an auth file), so the registry password
    # is never placed on a command line / visible in the process list.
    ecr_pass = subprocess.run(
        ["aws", "ecr", "get-login-password", "--region", region],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(
        ["skopeo", "login", "--username", "AWS", "--password-stdin", registry],
        input=ecr_pass, text=True, check=True,
    )

    for item in items:
        print(f"Publishing {item.image_name} -> {item.ecr_tag}")
        # skopeo is pre-installed on ubuntu-latest. The tar is
        # `podman save --format docker`, so use the docker-archive transport.
        _run(["skopeo", "copy", f"docker-archive:{item.oci_tar}", f"docker://{item.ecr_tag}"])
        _run(["aws", "s3", "cp", "--region", region, str(item.sif), item.s3_uri])

    print(f"Published {len(items)} image(s) for selector '{args.selector}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
