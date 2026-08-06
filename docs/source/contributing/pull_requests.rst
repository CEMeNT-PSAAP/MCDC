.. _pull_requests:

=============
Pull Requests
=============

MC/DC uses a fork-based contribution workflow. Open pull requests from your
fork against the ``dev`` branch of ``mcdc-project/mcdc``.

Before opening a pull request, make sure:

- The applicable tests pass.
- The code follows the project style.
- Tests and documentation have been added or updated as needed.
- ``CHANGELOG.md`` has been updated when the change is notable.

Changelog Updates
-----------------

Add a concise entry under ``[Unreleased]`` in
`CHANGELOG.md <https://github.com/mcdc-project/mcdc/blob/dev/CHANGELOG.md>`_
when a pull request introduces a notable user- or developer-visible change.
Follow the format described in that file. Release headings, versions, and dates
are assigned during release preparation.

Pull Request Description
------------------------

Use the pull request template to summarize:

- The type and purpose of the change.
- Associated issues or pull requests.
- Relevant theory or design context.
- New, changed, deprecated, or removed functionality.
- Any new dependency.
- The developers who should be notified.

The description should give reviewers enough context to understand the change,
verify its scope, and identify any follow-up work.
