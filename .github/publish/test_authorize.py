#!/usr/bin/env python3
"""Unit tests for the container-publish authorization decision.

Run: python3 -m unittest discover -s .github/publish -p 'test_*.py'

The decision logic is exercised with a FakeApi returning canned responses, so
these run offline and deterministically. Cases mirror the real GitHub API
behaviors verified live during development (annotated-tag peeling, PRs that
merely contain a commit vs. head it, empty pull lists, missing refs).
"""

from __future__ import annotations

import unittest

from authorize import Decision, Facts, decide, slugify

SHA = "c45e4ab4f2356589ba40f4fde2403f5854282826"
OTHER_SHA = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
REPO = "awslabs/palace"


class FakeApi:
    """Stand-in for GitHubApi. All lookups come from constructor data."""

    def __init__(self, *, refs=None, pr_for_head=None, labels=None):
        # refs: {("heads"|"tags", name): commit_sha}
        self._refs = refs or {}
        # pr_for_head: {head_sha: (number, head_repo_full_name, state)}
        self._pr = pr_for_head or {}
        # labels: {pr_number: set(label names)}
        self._labels = labels or {}

    def ref_sha(self, kind, name):
        return self._refs.get((kind, name))

    def pr_for_head(self, sha):
        return self._pr.get(sha)

    def pr_has_label(self, number, label):
        return label in self._labels.get(number, set())


def facts(**kw):
    base = dict(
        repo=REPO,
        event="push",
        head_sha=SHA,
        head_branch="main",
        head_repo=REPO,
        base_repo=REPO,
    )
    base.update(kw)
    return Facts(**base)


class ForkRejection(unittest.TestCase):
    def test_fork_head_repo_rejected_regardless_of_event(self):
        for event in ("push", "workflow_dispatch", "pull_request"):
            d = decide(facts(event=event, head_repo="attacker/palace"), FakeApi())
            self.assertFalse(d.authorized)
            self.assertIn("fork", d.reason)


class PushMain(unittest.TestCase):
    def test_main_at_built_sha_authorizes(self):
        api = FakeApi(refs={("heads", "main"): SHA})
        d = decide(facts(head_branch="main", head_sha=SHA), api)
        self.assertEqual(d, Decision(True, "main", d.reason))

    def test_stale_main_rejected(self):
        api = FakeApi(refs={("heads", "main"): OTHER_SHA})
        d = decide(facts(head_branch="main", head_sha=SHA), api)
        self.assertFalse(d.authorized)
        self.assertEqual(d.selector, "")


class ReleaseTag(unittest.TestCase):
    def test_release_tag_authorizes_and_strips_v(self):
        api = FakeApi(refs={("tags", "v0.18.0"): SHA})
        d = decide(facts(head_branch="v0.18.0", head_sha=SHA), api)
        self.assertTrue(d.authorized)
        self.assertEqual(d.selector, "0.18.0")

    def test_release_tag_wrong_sha_rejected(self):
        api = FakeApi(refs={("tags", "v0.18.0"): OTHER_SHA})
        d = decide(facts(head_branch="v0.18.0", head_sha=SHA), api)
        self.assertFalse(d.authorized)

    def test_nonexistent_tag_rejected(self):
        d = decide(facts(head_branch="v9.9.9", head_sha=SHA), FakeApi())
        self.assertFalse(d.authorized)

    def test_bad_syntax_tag_rejected(self):
        # ref exists at the sha but name is not strict vX.Y.Z
        api = FakeApi(refs={("tags", "v1.2"): SHA})
        d = decide(facts(head_branch="v1.2", head_sha=SHA), api)
        self.assertFalse(d.authorized)

    def test_feature_branch_push_rejected(self):
        d = decide(facts(head_branch="some-feature", head_sha=SHA), FakeApi())
        self.assertFalse(d.authorized)


class Dispatch(unittest.TestCase):
    def test_branch_at_built_sha_authorizes_moving_dev_selector(self):
        api = FakeApi(refs={("heads", "my/feature"): SHA})
        d = decide(facts(event="workflow_dispatch", head_branch="my/feature", head_sha=SHA), api)
        self.assertTrue(d.authorized)
        self.assertEqual(d.selector, "dev-my-feature")  # slash slugified

    def test_dispatch_on_tag_ref_rejected(self):
        # A tag isn't a branch, so heads/<name> resolves to nothing.
        api = FakeApi(refs={("tags", "v0.18.0"): SHA})
        d = decide(facts(event="workflow_dispatch", head_branch="v0.18.0", head_sha=SHA), api)
        self.assertFalse(d.authorized)

    def test_dispatch_stale_branch_rejected(self):
        api = FakeApi(refs={("heads", "b"): OTHER_SHA})
        d = decide(facts(event="workflow_dispatch", head_branch="b", head_sha=SHA), api)
        self.assertFalse(d.authorized)


class PullRequest(unittest.TestCase):
    def test_labeled_open_same_repo_pr_authorizes(self):
        api = FakeApi(pr_for_head={SHA: (840, REPO, "open")}, labels={840: {"push-containers"}})
        d = decide(facts(event="pull_request", head_branch="feat", head_sha=SHA), api)
        self.assertTrue(d.authorized)
        self.assertEqual(d.selector, "dev-feat")

    def test_unlabeled_pr_rejected(self):
        api = FakeApi(pr_for_head={SHA: (841, REPO, "open")}, labels={841: {"no-long-tests"}})
        d = decide(facts(event="pull_request", head_sha=SHA), api)
        self.assertFalse(d.authorized)
        self.assertIn("label", d.reason)

    def test_closed_but_labeled_pr_rejected(self):
        # A closed (or merged) PR must not publish even if it still carries the label.
        api = FakeApi(pr_for_head={SHA: (842, REPO, "closed")}, labels={842: {"push-containers"}})
        d = decide(facts(event="pull_request", head_sha=SHA), api)
        self.assertFalse(d.authorized)
        self.assertIn("not open", d.reason)

    def test_no_pr_heads_this_sha_rejected(self):
        # e.g. the sha is a main commit or only a member of a PR, not its head
        d = decide(facts(event="pull_request", head_sha=SHA), FakeApi(pr_for_head={}))
        self.assertFalse(d.authorized)
        self.assertIn("no PR", d.reason)

    def test_fork_pr_via_head_repo_rejected(self):
        # Same-repo base check passed, but the PR's own head repo is a fork.
        api = FakeApi(pr_for_head={SHA: (900, "attacker/palace", "open")}, labels={900: {"push-containers"}})
        d = decide(facts(event="pull_request", head_sha=SHA), api)
        self.assertFalse(d.authorized)
        self.assertIn("fork", d.reason)


class UnknownEvent(unittest.TestCase):
    def test_schedule_fails_closed(self):
        d = decide(facts(event="schedule"), FakeApi())
        self.assertFalse(d.authorized)
        self.assertIn("unsupported", d.reason)


class Slugify(unittest.TestCase):
    def test_slash_and_metachars_collapse(self):
        self.assertEqual(slugify("feature/a b"), "feature-a-b")
        self.assertEqual(slugify("ok.name_1-2"), "ok.name_1-2")  # allowed chars kept


if __name__ == "__main__":
    unittest.main()
