import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import statistics

def link_corrector(link):
    return link[5:-6]
print(link_corrector("<url>xcd32112.smart_meter.com</url>"))
