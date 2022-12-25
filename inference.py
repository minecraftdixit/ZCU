#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyxir
import tvm
from tvm.contrib import graph_executor
import numpy as np

dev = tvm.cpu()

# input_name = ...
# input_data = ...

# load the module into memory
lib = tvm.runtime.load_module("deploy_lib_edge.so")

module = graph_executor.GraphModule(lib["default"](dev))
module.set_input('input', np.zeros((1,3,1024,768)))
module.run()

