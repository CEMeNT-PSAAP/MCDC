.. _release_policy:

==========================
Release Policy and Process
==========================

MC/DC follows `Semantic Versioning <https://semver.org/>`_ and maintains a human-readable release history in `CHANGELOG.md <https://github.com/mcdc-project/mcdc/blob/dev/CHANGELOG.md>`_.

Minor Releases
--------------

MC/DC plans one minor release in each three-month seasonal cycle.
These releases collect compatible features, improvements, and fixes that have passed the project's required review and validation.
For planning convenience, these cycles follow the Northern Hemisphere meteorological seasons: winter (Dec-Feb), spring (Mar-May), summer (Jun-Aug), and autumn/fall (Sep-Nov).

The seasonal schedule is a target rather than a reason to release unverified work.
A minor release may be delayed when additional testing, documentation, or integration work is needed; it may also be brought forward when the accumulated changes are substantial and users would benefit from earlier availability.

Patch Releases
--------------

Bug fixes are not held until the next seasonal minor release.
Once a fix has passed review and the relevant validation and release checks, MC/DC publishes a patch release as soon as practical.

Published versions and release notes are available from the `MC/DC releases page <https://github.com/mcdc-project/mcdc/releases>`_.

.. _release_checklist:

Release Checklist
-----------------

Use this checklist for every minor and patch release.
A patch release may omit items that do not apply, but it must still complete dependency review, validation, and publication checks.

Prepare the Release
^^^^^^^^^^^^^^^^^^^

#. Confirm the intended version and scope against the release policy above.
#. Review the ``Unreleased`` section of ``CHANGELOG.md``.
   Ensure every user-visible change is included under the correct heading, remove empty headings, and add contributor attribution where appropriate.
#. Finalize the release version and date in ``CHANGELOG.md`` and ``CITATION.cff``, and update the stable entry's display name in ``docs/source/_static/switcher.json`` to the full ``X.Y.Z (stable)`` version while retaining ``stable`` as its version identifier and URL.
#. Confirm that documentation, examples, deprecation notices, and migration guidance match the release behavior.
#. Verify the supported Python versions in ``pyproject.toml``, continuous integration, and the user documentation agree.

Review and Consolidate Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Review every build, runtime, documentation, and development dependency in ``pyproject.toml``.
#. Check each dependency's current release, release notes, supported Python versions, and compatibility with MC/DC.
#. Consolidate the dependency declarations: remove unused or duplicate dependencies, keep shared constraints consistent, and ensure each direct dependency is declared in the appropriate group.
#. Set or update explicit lower bounds where MC/DC relies on a minimum feature and explicit upper bounds at the newest compatibility-tested release line.
   Do not widen a ceiling until the new line has passed the relevant unit, regression, documentation, and type-checking workflows.
#. Test the resolved environment for every supported Python version.
   Where practical, also test environments near the declared minimum and maximum bounds so that a successful default resolution does not hide an invalid constraint.
#. Record dependency additions, removals, or compatibility-bound changes in ``CHANGELOG.md``.

Validate the Release
^^^^^^^^^^^^^^^^^^^^

#. Run the formatter, unit tests, public API type checks, regression tests, and documentation build.
#. Confirm all required continuous-integration jobs pass on the release commit, including the manually triggered compatibility jobs for supported Python versions and applicable CPU, MPI, and GPU configurations.
#. Build the source distribution and wheel, then install and smoke-test both artifacts in clean environments.
#. Check the package metadata, bundled files, version, license, project links, and ``CITATION.cff``.

Integrate the Release
^^^^^^^^^^^^^^^^^^^^^

In this workflow, ``upstream`` refers to the canonical ``mcdc-project/mcdc`` repository and ``make_release`` refers to the release-preparation branch.

#. Merge the completed ``make_release`` branch into ``upstream/dev`` through a reviewed pull request.
#. After the release candidate passes the required checks, merge ``upstream/dev`` into ``upstream/main`` through a reviewed pull request.
#. Use the resulting ``upstream/main`` commit as the release base.

Publish from Main
^^^^^^^^^^^^^^^^^

#. Create the ``v``-prefixed tag and GitHub release from the validated release commit on ``upstream/main``.
#. Confirm that the package, citation-metadata, and stable-documentation publication workflows succeed.
#. Install the published package from PyPI in a clean environment and run a minimal MC/DC simulation.

Return to Development
^^^^^^^^^^^^^^^^^^^^^

#. Merge ``upstream/main`` back into ``upstream/dev`` after the release is published and verified.
#. Prepare ``upstream/dev`` for the next development cycle and confirm its required checks pass.
#. Remove the merged ``make_release`` branch when it is no longer needed.
