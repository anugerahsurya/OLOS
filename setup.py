from setuptools import setup, find_packages

# Baca isi requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    
setup(
    name='OLOS',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='Anugerah Surya Atmaja',
    author_email='atmajasuryaanugerah@gmail.com',
    description='Package ini berguna untuk membantu proses Data Preparation pada Analisis Data dengan Python. Pada package ini juga ditambahkan kode untuk menemukan best model untuk sebuah kasus spesifik terkait hybrid Regresi dan Klasifikasi.',
    url='https://github.com/anugerahsurya/OLOS',
)
