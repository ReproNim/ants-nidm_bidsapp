"""ANTs-backed segmentation for the ants-nidm BIDS App.

`ANTsSegmentation` is imported lazily so that the light submodules -- currently
`label_vocabulary`, which is pure numpy -- stay importable without ANTsPy.

This is not only about CI. `tests/test_wrapper.py` installs a mock into
`sys.modules['ants']` and then imports the wrapper, which only works while no
earlier import has already bound the real ANTsPy into `src.antspy.wrapper`.
When this module imported the wrapper eagerly, importing *any* submodule of
`src.antspy` first was enough to silently defeat that mock, and the wrapper
tests failed inside ANTsPy with a `ptrstr()` type error rather than anywhere
that pointed at the cause. `src/registration/__init__.py` already avoids the
same trap the same way.
"""

__all__ = ['ANTsSegmentation']


def __getattr__(name):
    if name == 'ANTsSegmentation':
        from src.antspy.wrapper import ANTsSegmentation

        return ANTsSegmentation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
