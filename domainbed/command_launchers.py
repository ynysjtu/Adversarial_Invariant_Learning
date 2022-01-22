# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import os
import asyncio
import sys

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

async def run_one(cmd, sem, i):
    cmd = "CUDA_VISIBLE_DEVICES=" + str(i%8) + " " + cmd
    async with sem:
        proc = await asyncio.create_subprocess_shell(cmd)
        await proc.wait()

async def launch_cluster(commands):
    sem = asyncio.Semaphore(8)
    await asyncio.gather(*[run_one(cmd, sem, i) for i, cmd in enumerate(commands)])

def cluster_launcher(commands):
    """Launch commands on a 8 GPU machine"""
    asyncio.run(launch_cluster(commands))

def dummy_launcher(commands):
    """Doesn't run anything; instead, prints each command.
    Useful for testing."""
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'cluster': cluster_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
