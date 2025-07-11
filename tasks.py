"""Tasks to release the package."""

import os
import shutil
import sys

import pytest

if sys.version_info.minor < 11:
    import tomli as tomllib
else:
    import tomllib
from invoke import Collection, Exit, task

PYTHON = "python"
ns = Collection()


test_collection = Collection("test")


@task
def unittest(c):
    """Run unittests."""
    import micromagnetictests

    result = micromagnetictests.test()
    raise Exit(code=result)


@task
def coverage(c):
    """Run unittests with coverage."""
    result = pytest.main(["-v", "--cov", "micromagnetictests", "--cov-report", "xml"])
    raise Exit(code=result)


@task
def docs(c):
    """Run doctests."""
    result = pytest.main(
        [
            "-v",
            "--doctest-modules",
            "--ignore",
            "micromagnetictests/tests",
            "micromagnetictests",
        ]
    )
    raise Exit(code=result)


@task
def ipynb(c):
    """Test notebooks."""
    result = pytest.main(["-v", "--nbval", "--sanitize-with", "nbval.cfg", "docs"])
    raise Exit(code=result)


@task
def all(c):
    """Run all tests."""
    for cmd in (unittest, docs, ipynb):
        try:
            cmd(c)
        except Exit as e:
            if e.code != pytest.ExitCode.OK:
                raise e
    raise Exit(code=pytest.ExitCode.OK)


test_collection.add_task(unittest)
test_collection.add_task(coverage)
test_collection.add_task(docs)
test_collection.add_task(ipynb)
test_collection.add_task(all)
ns.add_collection(test_collection)


@task
def build_dists(c):
    """Build sdist and wheel."""
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    c.run(f"{PYTHON} -m build")
    c.run("twine check dist/*")


@task(build_dists)
def upload(c):
    """Upload package to PyPI."""
    c.run("twine upload dist/*")


@task
def release(c):
    """Run the whole release process.

    Steps:
    - Pull all changes in ``master``.
    - Tag last commit using version information from setup.cfg/pyproject.toml.
    - Update the ``latest`` tag to point to the last commit.
    - Build package (``sdist`` and ``wheel``).
    - Upload package to PyPI.
    - Push new tags.
    """
    c.run("git checkout master")
    c.run("git pull")

    res = c.run("git status -s", hide=True)
    if res.stdout != "":
        raise Exit("Working tree is not clean. Aborting.")

    # run all tests
    try:
        all(c)
    except Exit as e:
        if e.code != pytest.ExitCode.OK:
            raise e

    with open("pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]

    c.run(f"git tag {version}")  # fails if the tag exists
    c.run("git tag -f latest")  # `latest` tag for binder

    build_dists(c)
    upload(c)

    c.run("git push -f --tags")
    c.run("git push")


ns.add_task(build_dists)
ns.add_task(upload)
ns.add_task(release)
