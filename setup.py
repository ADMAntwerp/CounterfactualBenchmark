from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

with open('LICENSE.txt') as f:
    LICENSE = f.read()

with open('HISTORY.md') as f:
    HISTORY = f.read()

setup_args = dict(
    name='cfbench',
    version='0.0.7',
    description='Benchmarking tool for Counterfactual Explanations',
    long_description_content_type='text/markdown',
    long_description=README + '\n\n' + HISTORY,
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    author='Raphael Mazzine Barbosa de Oliveira',
    keywords=['Counterfactual Explanations', 'Benchmarking', 'Machine Learning'],
    url='https://github.com/ADMAntwerp/CounterfactualBenchmark',
    download_url='https://pypi.org/project/cfbench/',
)

install_requires = [
    'pandas',
    'tensorflow',
    'requests',
    'scipy'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)