import setuptools
from setuptools import setup
from hpcp import __version__

classifiers = [
    'Programming Language :: Python :: 3',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Microsoft :: Windows',
]
# setuptools 77+ (PEP 639) takes an SPDX license expression and warns about license classifiers (an error next to an
# expression); older ones (the newest for Python 3.8 and earlier, so all a Python 3.6 install has) know only the
# free-text license and the classifier.
if int(setuptools.__version__.split('.')[0]) >= 77:
    license_metadata = {'license_expression': 'GPL-3.0-or-later', 'license_files': ['LICENSE']}
else:
    license_metadata = {'license': 'GPL-3.0-or-later'}
    classifiers.append('License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)')

setup(
    name='hpcp',
    version=__version__,
    description='Highly Parallel CoPy / HPC coPy: A simple script optimized for distributed file store / NVMe / SSD storage medias for use in High Performace Computing environments.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='Yufei Pan',
    author_email='pan@zopyr.us',
    url='https://github.com/yufei-pan/hpcp',
    py_modules=['hpcp'],
    entry_points={
        'console_scripts': [
            'hpcp=hpcp:main',
        ],
    },
    install_requires=[
		# not on 3.6: stock pip on EL7 and EL8 takes no xxhash wheel, and building one needs a compiler; hpcp runs without it
		"xxhash; python_version >= '3.7'",
		# No multiCMD: hpcp carries its own copy (the embedded block in hpcp.py) and runs it wherever an installed one is
		# older than that copy, missing or does not import, so a requirement would guarantee nothing; and pip 9 (stock on EL7
		# and EL8) upgrading an old multiCMD for it stopped `pip3 install -U hpcp` (docs/lessons.md, 2026-11-11).
    ],
    classifiers=classifiers,
    python_requires='>=3.6',
    **license_metadata
)
