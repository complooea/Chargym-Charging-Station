from setuptools import setup

setup(name='Chargym_Charging_Station',
      version='0.0.1',
      packages=['Chargym_Charging_Station'],
      install_requires=[
            "setuptools==59.5.0",
            "wheel>=0.36,<0.40",
            "stable-baselines3[extra]==1.4.0"
      ],
)