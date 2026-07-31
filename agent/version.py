"""Single source of truth for the app's version label.

Every user-visible version string resolves here — the sidebar "Console vX.Y.Z",
the Settings footer, the ``/api/health`` payload the iOS client reads, and the
README badge — so the app can never label itself with a version that was never
released. Before this existed the number was typed out in four places and all
four had drifted to 2.2.0 while the shipped release was 2.4.0.

The number is baked in rather than read from git at runtime, deliberately: a
copy without git metadata (zip download, packaged install) must still label
itself correctly, and two machines running identical code must agree regardless
of which tags each happens to have fetched.

What makes it *stay* true is ``tests/test_app_version.py``: it fails the suite
when this constant drifts from the newest release tag or from the README badge.
So cutting a release is ``git tag vX.Y.Z`` **and** bumping this line — forget
the second and the tests say so.
"""

__version__ = "2.4.0"
