.. _documentation_sphinx:

.. highlight:: none

Sphinx and Read the Docs
========================

MC/DC uses `Sphinx <https://www.sphinx-doc.org/>`_ to generate its documentation and `Read the Docs <https://about.readthedocs.com/>`_ to build and host the documentation website.

This page introduces the subset of Sphinx needed to contribute to MC/DC's documentation, including reStructuredText, document organization, automatic API generation, and local documentation builds.


reStructuredText and Sphinx
---------------------------

We write files for Sphinx using a plaintext markup language called reStructuredText (rst).
`Click here for a rst Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.
Sphinx builds an html file for every rst file in the documentation root directory and its subdirectories our documentation root directory is ``mcdc/docs/source/``.
The root document, ``index.rst``, serves as the welcome page.
The root directory also contains task-oriented sections, each with an ``index.rst`` landing page and related topic pages.
It's useful to compare our rst files to their associated webpages to get a feel for how they translate.

Source Formatting
^^^^^^^^^^^^^^^^^

Documentation source follows a one-sentence-per-line convention.
Start each new sentence on a new physical line, and do not wrap a sentence to meet a fixed line length.
This semantic line structure makes changes easier to review and keeps unrelated sentences out of the same diff.
Blank lines still separate paragraphs, and indentation required by reStructuredText lists, directives, and other structured blocks must be preserved.
Code blocks, tables, generated content, URLs, and other syntax that requires a particular layout are exempt from this convention.


Like any plaintext markup language, rst uses "explicit markup" for constructs that need special handling, such as including a code-block or cross-referencing other pages.

A block of explicit markup text starts with ".. " and is terminated by the next paragraph at the same level of indentation.

Sphinx creates webpage elements using explicit markup blocks called directives.

.. tip::
   For example, this block was created using the `tip` directive!

::

  .. tip::
     For example, this block was created using the `tip` directive!

An explicit markup block without a directive is taken as a comment that will not appear on the webpage: ::

  .. For example, this is a comment.

In addition to directives for blocks of explicit markup, Sphinx handles in-line explicit markup with roles.
For example, this equation :math:`a^2 + b^2 = c^2` was created using the `math` role.
::

  For example, this equation :math:`a^2 + b^2 = c^2` was created using the `math` role.

`Click here for a list of Sphinx directives <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html>`_ and `click here for a list of Sphinx roles <https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html>`_.


The toctree
-----------

Sphinx's main directive is the `toctree` directive, which generates a table of contents tree (toctree) with links to other webpages in the build.
The listed documents should be named relative to the current document and excluding the .rst extension.
For example, the following on ``index.rst`` creates a table of contents on the main page with links to the main user-facing sections: ::

  .. toctree::
     user_guide/index
     theory/index
     examples/index

Sphinx will build an html file for all rst files in the source directory and its subdirectories.
Sphinx will issue a warning if an html file isn't referenced in any toctree because that means that the generated webpage is not reachable through standard navigation.


Using autodoc and autosummary
-----------------------------

Within MC/DC's source code, we document functions and classes using docstrings.
`We use two Sphinx extensions <https://romanvm.pythonanywhere.com/post/autodocumenting-your-python-code-sphinx-part-ii-6/>`_ -- ``autodoc`` and ``autosummary`` -- to generate rst files for Sphinx using the existing docstrings in our source code.
For ``autodoc`` and ``autosummary`` to work, the docstrings within MC/DC's source code must be written in correct rst.

The ``autodoc`` extension includes a set of directives to document different chunks of code (e.g., modules, functions, classes).
For example, below is the entire rst file that generates the :doc:`../../reference/python_api/generated/mcdc.NeutronMultigroupData` page:

.. code-block::

   mcdc.NeutronMultigroupData
   ======================

   .. currentmodule:: mcdc

   .. autoclass:: NeutronMultigroupData

(That in-line reference was created using :code:`:doc:\`../../reference/python_api/generated/mcdc.NeutronMultigroupData\``, by the way).

A rst file with an ``autodoc`` directive is required for each module or function that we would like to document.
Rather than create all of these rst files by hand, we use the ``autosummary`` extension to do it for us.

For example, consider the first ``autosummary`` directive in ``source/reference/python_api/index.rst``, the file that governs the :doc:`../../reference/python_api/index` page:

.. code-block::

   .. autosummary::

      mcdc.Material
      mcdc.NeutronMultigroupData

This directive:
  #. Generates two files in ``reference/python_api/generated/``: ``mcdc.Material.rst`` and ``mcdc.NeutronMultigroupData.rst``.
  #. Populates each file with the proper autoclass directive.
  #. Creates a table on :doc:`../../reference/python_api/index` with entries mcdc.Material and mcdc.NeutronMultigroupData that link to the respective generated pages.


Building Locally
----------------

We can check our work with a local build.
Make sure you're in ``mcdc/docs/``:

#. Both Sphinx and the PyData Sphinx Theme should have been installed with MC/DC.
   To check, type ``sphinx-build --version`` on the commandline.
   If they are not installed, run ``pip install -e ".[docs]"`` from the repository root.
#. With Sphinx installed, run ``make html``.
   This builds local html files in ``mcdc/docs/build/``.
#. To launch your local html from the commandline, ``open build/html/index.html``.
   Check your work: has your content been added or changed as you expected?
#. Continue making changes to your local rst files, building locally, and launching the built html files until you're satisfied with how the website will look.

The API reference imports MC/DC during the documentation build, so run a complete local build after changing package imports or dependencies.

Once you are satisfied with the local build, submit a PR.
