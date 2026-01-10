"""Setup script for PaperShield package."""

from setuptools import find_packages, setup

setup(
    name="papershield",
    version="0.1.0",
    description="Measuring & Mitigating Authority-Context Prompt Injection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "papershield-run=scripts.run_eval:main",
            "papershield-report=scripts.make_report:main",
        ],
    },
)

