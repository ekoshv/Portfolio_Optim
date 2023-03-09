from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ekoptim",
    version="0.0.17",
    packages=["ekoptim"],
    url="https://github.com/ekoshv/Markowitz_Optim",
    project_urls={
        "LinkedIn": "https://www.linkedin.com/in/ehsan-khademolama-05149865/",
    },
    license="Licensed",
    author="@EhsanKhademOlama",
    author_email="ekoshv.igt@gmail.com",
    description="Portfolio Optimizers",
    long_description_content_type="text/markdown",
    long_description=long_description,
    install_requires=[
        "setuptools",
        "pandas",
        "scipy",
        "sklearn"
    ],
)
