#!/usr/bin/env python3
"""
Test script for the visualization module
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import from the new location in Experiments folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'Experiments'))
from visualize import main

if __name__ == "__main__":
    # This will use the default config from Experiments/config/visualize.yaml
    main() 