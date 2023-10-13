# Ignore certain packages conda does
pip list --format=freeze | grep -Po '^((?!mkl\-fft).)*$' >requirements.txt
