import numpy as np
import pandas as pd

def link_corrector(link):
    return link[12:-6]
print(link_corrector("<url>xcd32112.smart_meter.com</url>"))
