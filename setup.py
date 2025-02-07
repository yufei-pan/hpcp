from setuptools import setup
from hpcp import __version__

setup(
    name='hpcp',
    version=__version__,
    description='Highly Parallel CoPy / HPC coPy: A simple script optimized for distributed file store / NVMe / SSD storage medias for use in High Performace Computing environments.',
    long_description=open('README.md').read(),
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
        'argparse',
		'xxhash',
		'multiCMD>1.19',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: POSIX :: Linux',
		'Operating System :: Microsoft :: Windows',
    ],
    python_requires='>=3.6',
	license='GPLv3+',
)
