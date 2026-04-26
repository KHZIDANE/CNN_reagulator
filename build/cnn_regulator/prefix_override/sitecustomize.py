import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/zidane/Desktop/project_cnn_regulation/install/cnn_regulator'
