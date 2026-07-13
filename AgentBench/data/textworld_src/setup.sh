#!/usr/bin/env bash

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set -ex
echo "Running setup.sh...";

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGW;;
    *)          machine="UNKNOWN:${unameOut}"
esac

cd textworld/thirdparty/

# Inform 7 CLI download skipped (network blocked); ALFWorld uses pre-compiled data.

# Install modified fork of Glulx and build it - we have this one locally
(
    echo "Installing cheapglk"
    cd glulx/cheapglk
    make -B
)
(
    echo "Installing git-glulx-ml"
    cd glulx/Git-Glulx
    make -B
)
