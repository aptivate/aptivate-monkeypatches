from setuptools import setup

long_description = '''\
Aptivate monkey patches - specific patches useful in Django projects.'''

setup(
    author="Chris Wilson",
    author_email="chris-monkeypatches@aptivate.org",
    name='aptivate-monkeypatches',
    version='1.0',
    description='Monkeypatch tools',
    long_description=long_description,
    url='https://github.com/aptivate/aptivate-monkeypatches/',
    platforms=['OS Independent'],
    license='MIT License',
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Utilities',
    ],
    include_package_data=True,
    zip_safe=False
)
