from setuptools import setup

# TODO

with open("README.md", "r") as r:
    long_description = r.read()

setup(
    name='SynRaptor',
    version='0.0.1',
    description='TODO',     # TODO
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy >= 1.15',
                      'scipy >= 1.3',
                      'matplotlib >= 3'],
    package_dir={'': 'src'}
)
