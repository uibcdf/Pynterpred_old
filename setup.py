import distutils.extension
from setuptools import setup, find_packages

setup(
    name='pynterpred',
    version='0.1.1',
    author='UIBCDF Lab',
    author_email='uibcdf@gmail.com',
    package_dir={'pynterpred': 'pynterpred'},
    packages=find_packages(),
    package_data={'pynterpred': []},
    scripts=[],
    url='http://uibcdf.org',
    download_url ='https://github.com/uibcdf/Pynterpred',
    license='MIT',
    description="---",
    long_description="---",
)
