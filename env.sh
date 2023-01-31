#!/bin/bash
export CNINDEX_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CNINDEX_PATH}/lib 
