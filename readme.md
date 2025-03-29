This is what happens when you want a shared codebase between multiple repos but don't want to have to rebuild it each time... thank god for setuptools --editable

To use in editable mode, first clone the repository, cd into it, activate the virtual/conda environment you want it in, then do `pip install -e .`

It just works!