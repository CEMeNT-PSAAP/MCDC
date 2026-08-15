.. _release_policy:

==========================
Release Policy and Process
==========================

MC/DC uses `Semantic Versioning <https://semver.org/>`_ as a guide for distinguishing minor and patch releases and maintains a human-readable release history in `CHANGELOG.md <https://github.com/mcdc-project/mcdc/blob/dev/CHANGELOG.md>`_.

Minor Releases
--------------

For MC/DC, a minor release primarily expands what the code can do.
Examples include an additional transport or execution capability, a new public Python API, a new command-line option, a new opt-in behavior, or a deprecation that users need time to accommodate.
A minor release may include compatible fixes and internal improvements alongside its features.

MC/DC plans one minor release in each three-month seasonal cycle.
These releases collect compatible features, improvements, and fixes that have passed the project's required review and validation.
For planning convenience, these cycles follow the Northern Hemisphere meteorological seasons: winter (Dec-Feb), spring (Mar-May), summer (Jun-Aug), and autumn/fall (Sep-Nov).

The seasonal schedule is a target rather than a reason to release unverified work.
A minor release may be delayed when additional testing, documentation, or integration work is needed; it may also be brought forward when the accumulated changes are substantial and users would benefit from earlier availability.

Patch Releases
--------------

For MC/DC, a patch release primarily restores or strengthens behavior already expected from the current stable release.
A change belongs in a patch when it corrects intended or documented behavior, prevents an installation or runtime failure, or fixes an implementation defect without asking users to adopt a new interface.
Tests, documentation corrections, refactoring, build changes, and developer tooling may accompany a patch when they support or validate the fix and do not independently expand user-facing behavior.
If a change introduces a new option, API, capability, or opt-in behavior, it should normally be held for a minor release even when developed alongside a bug fix and separated from the patch branch.
When the classification is not obvious, use the purpose and user impact of the release: capability expansion points to a minor release, while correction of existing behavior points to a patch release.

Bug fixes are not held until the next seasonal minor release.
Once a fix has passed review and the relevant validation and release checks, MC/DC publishes a patch release as soon as practical.
Every patch release must include a non-empty ``Fixed`` section in ``CHANGELOG.md`` that describes the user-visible defect corrected by the release.
Place ``Fixed`` first in a patch release entry so the reason for the release is immediately visible.
Supporting changes may also appear under other headings, but the defect that justifies the patch release must be stated under ``Fixed``.
For dependency-compatibility patches, identify the installation or runtime failure prevented and the affected dependency or version range when known.

Published versions and release notes are available from the `MC/DC releases page <https://github.com/mcdc-project/mcdc/releases>`_.

.. _release_branch_routes:

Release Branch Routes
---------------------

MC/DC uses different integration routes for feature and patch releases because ``upstream/dev`` and ``upstream/main`` have different roles.
The development branch integrates work for the next minor release, while the main branch identifies the current stable release line.

.. list-table:: Release workflow comparison
   :header-rows: 1
   :widths: 24 24 28 44

   * - Workflow
     - Release branch base
     - Route to release
     - Purpose
   * - Feature (minor-release) workflow
     - ``upstream/dev``
     - release branch → ``upstream/dev`` → ``upstream/main``
     - Includes the compatible features, improvements, and fixes accumulated for the next minor version.
   * - Patch workflow
     - ``upstream/main``
     - patch branch → ``upstream/main``
     - Releases selected fixes from the current stable line without including unreleased features already present on ``upstream/dev``.

If a patch fix was first developed on ``upstream/dev``, transfer only the fix and its necessary tests, documentation, and supporting changes onto a branch based on ``upstream/main``.
Do not merge ``upstream/dev`` into the patch branch.
After publishing the patch, merge ``upstream/main`` back into ``upstream/dev`` so subsequent minor releases retain the fix.

.. _release_checklist:

Release Checklist
-----------------

Use this checklist for every minor and patch release.
A patch release may omit items that do not apply, but it must still complete dependency review, validation, and publication checks.

Prepare the Release
^^^^^^^^^^^^^^^^^^^

#. Confirm the intended version and scope against the release policy above.
#. Select the appropriate :ref:`release_branch_routes` and create the release-preparation branch from its prescribed base.
#. Review the ``Unreleased`` section of ``CHANGELOG.md``.
   Ensure every user-visible change is included under the correct heading, remove empty headings, and add contributor attribution where appropriate.
   For a patch release, place ``Fixed`` first and confirm that it is non-empty and clearly states the defect that justifies the release.
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

Here, ``upstream`` refers to the canonical ``mcdc-project/mcdc`` repository and ``make_release`` refers to the release-preparation branch.

For a minor release using the feature workflow:

#. Create ``make_release`` from ``upstream/dev`` and complete the release preparation and validation there.
#. Merge the completed ``make_release`` branch into ``upstream/dev`` through a reviewed pull request.
#. After the release candidate passes the required checks, merge ``upstream/dev`` into ``upstream/main`` through a reviewed pull request.
#. Use the resulting ``upstream/main`` commit as the release base.

For a patch release using the dev-bypass workflow:

#. Create ``make_release`` from the current stable commit on ``upstream/main`` and include only the selected fixes and necessary supporting changes.
#. Merge the completed ``make_release`` branch directly into ``upstream/main`` through a reviewed pull request, without routing it through ``upstream/dev``.
#. Use the resulting ``upstream/main`` commit as the release base.

Publish from Main
^^^^^^^^^^^^^^^^^

#. Create the ``v``-prefixed tag and GitHub release from the validated release commit on ``upstream/main``.
#. Confirm that the package, citation-metadata, and stable-documentation publication workflows succeed.
#. Install the published package from PyPI in a clean environment and run a minimal MC/DC simulation.

Return to Development
^^^^^^^^^^^^^^^^^^^^^

#. Merge ``upstream/main`` back into ``upstream/dev`` after the release is published and verified.
#. For a patch release, confirm that the back-merge retains the patch while preserving the unreleased feature work already on ``upstream/dev``.
#. Prepare ``upstream/dev`` for the next development cycle and confirm its required checks pass.
#. Remove the merged ``make_release`` branch when it is no longer needed.
