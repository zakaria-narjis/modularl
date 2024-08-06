Installation Guide
==================

To install the package, use `pip`, the Python package installer. You can install the package along with its dependencies using the following command:

.. code-block:: bash

    pip install modularl

Dependencies
------------

The package has the following dependencies:

- **numpy**: Version 1.24 or higher
- **torch**: Version 2.0 or higher
- **torchrl**: Version 0.4.0 or higher
- **tensorboard**: Version 2.17.0 or higher

These dependencies will be automatically installed when you install the package using the command above.

Verifying the Installation
--------------------------

After installation, you can verify that the package and its dependencies are correctly installed by running:

.. code-block:: bash

    python -c "import modularl; import numpy; import torch; import torchrl; import tensorboard; print('Installation successful')"

If the installation was successful, you should see "Installation successful" printed to the console.

Troubleshooting
---------------

If you encounter issues during installation, ensure that:

- You have the correct version of Python installed.
- You are using the latest version of `pip`.
- Your network connection is stable.

For further assistance, please refer to the package documentation or seek help on the package's issue tracker.

