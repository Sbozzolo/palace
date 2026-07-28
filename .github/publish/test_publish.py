#!/usr/bin/env python3
"""Unit tests for the container-publish plan (pure logic; no I/O).

Run: python3 -m unittest discover -s .github/publish -p 'test_*.py'
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from publish import PublishItem, arch_label_from_image, plan, s3_prefix_for

REGISTRY = "123456789012.dkr.ecr.us-west-2.amazonaws.com"
ECR_REPO = "palace"
BUCKET = "palace-ci-containers"


def make_artifacts(root: Path, image_names: list[str]) -> None:
    """Create the downloaded-artifact layout: <image>-oci/<image>.tar and
    <image>-sif/<image>.sif for each image name."""
    for name in image_names:
        (root / f"{name}-oci").mkdir(parents=True)
        (root / f"{name}-oci" / f"{name}.tar").write_text("tar")
        (root / f"{name}-sif").mkdir(parents=True)
        (root / f"{name}-sif" / f"{name}.sif").write_text("sif")


class ArchLabel(unittest.TestCase):
    def test_native_label(self):
        self.assertEqual(arch_label_from_image("palace-sapphirerapids-abc1234"), "sapphirerapids")

    def test_release_target_with_underscores(self):
        self.assertEqual(arch_label_from_image("palace-x86_64_v3-deadbee"), "x86_64_v3")

    def test_rejects_non_palace(self):
        with self.assertRaises(ValueError):
            arch_label_from_image("notpalace-x-abc1234")

    def test_rejects_missing_hash(self):
        with self.assertRaises(ValueError):
            arch_label_from_image("palace-onlylabel")


class S3Prefix(unittest.TestCase):
    def test_main(self):
        self.assertEqual(s3_prefix_for("main"), "main")

    def test_release(self):
        self.assertEqual(s3_prefix_for("0.18.0"), "0.18.0")

    def test_dev_expands_to_slash(self):
        self.assertEqual(s3_prefix_for("dev-myfeature"), "dev/myfeature")


class Plan(unittest.TestCase):
    def test_release_matrix_six_targets(self):
        names = [f"palace-{t}-abc1234" for t in
                 ("x86_64_v3", "x86_64_v4", "sapphirerapids", "aarch64", "neoverse_v1", "neoverse_v2")]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_artifacts(root, names)
            items = plan(root, "0.18.0", REGISTRY, ECR_REPO, BUCKET)
        self.assertEqual(len(items), 6)
        by_label = {i.arch_label: i for i in items}
        self.assertEqual(
            by_label["neoverse_v2"].ecr_tag,
            f"{REGISTRY}/palace:0.18.0-neoverse_v2",
        )
        self.assertEqual(
            by_label["neoverse_v2"].s3_uri,
            f"s3://{BUCKET}/0.18.0/neoverse_v2.sif",
        )

    def test_dev_selector_s3_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_artifacts(root, ["palace-sapphirerapids-abc1234"])
            items = plan(root, "dev-myfeature", REGISTRY, ECR_REPO, BUCKET)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].ecr_tag, f"{REGISTRY}/palace:dev-myfeature-sapphirerapids")
        self.assertEqual(items[0].s3_uri, f"s3://{BUCKET}/dev/myfeature/sapphirerapids.sif")

    def test_empty_artifacts_yields_empty_plan(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(plan(Path(d), "main", REGISTRY, ECR_REPO, BUCKET), [])

    def test_item_points_at_real_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_artifacts(root, ["palace-aarch64-abc1234"])
            item = plan(root, "main", REGISTRY, ECR_REPO, BUCKET)[0]
        self.assertTrue(item.oci_tar.name.endswith(".tar"))
        self.assertTrue(item.sif.name.endswith(".sif"))
        self.assertIsInstance(item, PublishItem)


if __name__ == "__main__":
    unittest.main()
