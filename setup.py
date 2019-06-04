from setuptools import setup

dist = setup(name="Pmesh_emulator",
             author="Thomas McClintock",
             author_email="mcclintock@bnl.gov",
             description="Framework for emulating particle mesh simulations.",
             license="MIT",
             url="https://github.com/tmcclintock/Pmesh_emulator",
             include_package_data = True,
             packages=['pmesh_emulator'],
             long_description=open("README.md").read())
